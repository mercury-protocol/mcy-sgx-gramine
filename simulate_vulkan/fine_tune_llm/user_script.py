import evaluate
import numpy as np

import torch

from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    TrainerCallback,
    Trainer
)

STATE_DICT_READY_PATH = "state_dict_ready.pth"
GRADIENT_PATH = "gradient.pth"

class VulkanCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    # TODO: add the rest of the stuff from mcy_dist_ai
         
    def save_gradients(self):
        gradient = {name: param.data for name, param in self.model.named_parameters() if param.requires_grad}
        torch.save(gradient, GRADIENT_PATH)

    def on_step_end(self, args, state, control, **kwargs):
        self.save_gradients()

def main():
    # -------------------- DATA --------------------
    # load dataset
    dataset = load_dataset('shawhin/imdb-truncated')

    # -------------------- MODEL --------------------
    model_checkpoint = 'distilbert-base-uncased'

    # define label maps
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative": 0, "Positive": 1}

    # generate classification model from model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )

    # -------------------- PREPROCESS DATA --------------------
    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # create tokenize function
    def tokenize_function(examples):
        # extract text
        text = examples["text"]

        # tokenize and truncate text
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512
        )

        return tokenized_inputs

    # tokenize training and validation datasets
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------- EVALUATION --------------------
    # import accuracy evaluation metric
    accuracy = evaluate.load("accuracy")

    # define an evaluation function to pass into trainer later
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)

        return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

    # -------------------- TRAIN MODEL --------------------
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['q_lin']
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # hyperparameters
    lr = 1e-3
    batch_size = 4
    num_epochs = 1

    # define training arguments
    training_args = TrainingArguments(
        output_dir=model_checkpoint + "-lora-text-classification",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # create trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
        compute_metrics=compute_metrics,
    )

    callback = VulkanCallback(model=model)

    trainer.add_callback(callback)

    # train model
    trainer.train()


if __name__ == "__main__":
    main()
