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
    Trainer, TrainerState, TrainerControl
)


MODEL_CHECKPOINT = 'distilbert-base-uncased'

STATE_DICT_READY_PATH = "state_dict_ready.pth"
GRADIENT_PATH = "gradient.pth"


class VulkanCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    # TODO: add the rest of the stuff from mcy_dist_ai
         
    def save_gradients(self):
        gradient = {name: param.data for name, param in self.model.named_parameters() if param.requires_grad}
        torch.save(gradient, GRADIENT_PATH)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # self.model = load_network(path=STATE_DICT_PATH, delete_file=True)
        # optimizer = load_optimizer(network)
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # self.save_gradients()
        # checkpoint(epoch=state.epoch, batch_idx=batch_idx)
        #
        # if state.global_step == state.max_steps:
        #     self.signal_worker_finished()
        # else:
        #     await self.wait_state_dict()
        print(state.epoch, state.global_step, state.max_steps)


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


def main():
    tokenizer = create_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1}
    )

    if tokenizer.pad_token is None:
        # TODO: this step couples model and tokenizer, but seems like code execution doesn't enter here
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = create_dataset(tokenizer)
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
        output_dir=MODEL_CHECKPOINT + "-lora-text-classification",
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
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
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
