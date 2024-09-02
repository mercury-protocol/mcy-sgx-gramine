import evaluate
import numpy as np

from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


LEARNING_RATE = 1e-3
BATCH_SIZE = 4
N_EPOCHS = 1
MODEL_CHECKPOINT = 'distilbert-base-uncased'

accuracy = evaluate.load("accuracy")


def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
    tokenizer.truncation_side = "left"

    return tokenizer


def create_dataset(tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_tensors="np",
            truncation=True,
            max_length=512
        )

    raw_dataset = load_dataset('shawhin/imdb-truncated')
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1}
    )
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['q_lin']
    )

    return get_peft_model(model, peft_config)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}


def create_trainer():
    tokenizer = create_tokenizer()
    model = create_model()

    if tokenizer.pad_token is None:
        # TODO: this step couples model and tokenizer, but seems like code execution doesn't enter here
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = create_dataset(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------- TRAIN MODEL --------------------
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=MODEL_CHECKPOINT + "-lora-text-classification",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=N_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    # {
    #   'eval_loss': 0.3110983669757843,
    #   'eval_accuracy': {
    #     'accuracy': 0.901
    #   },
    #   'eval_runtime': 81.4019,
    #   'eval_samples_per_second': 12.285,
    #   'eval_steps_per_second': 3.071,
    #   'epoch': 1.0
    # }
    # {
    #   'train_runtime': 219.7641,
    #   'train_samples_per_second': 4.55,
    #   'train_steps_per_second': 1.138,
    #   'train_loss': 0.41538427734375,
    #   'epoch': 1.0
    # }

    trainer = create_trainer()
    trainer.train()
