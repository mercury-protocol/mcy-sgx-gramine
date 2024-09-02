import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    )


UNTRAINED_CHECKPOINT = 'distilbert-base-uncased'
TRAINED_CHECKPOINT = 'distilbert-base-uncased-lora-text-classification/checkpoint-250'


id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}


def load_model(checkpoint=TRAINED_CHECKPOINT):
    # generate classification model from model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )

    return model


def evaluate_model(model, checkpoint=TRAINED_CHECKPOINT):
    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # define list of examples
    text_list = [
        "It was good.",
        "Not a fan, don't recommend.",
        "Better than the first one.",
        "This is not worth watching even once.",
        "This one is a pass."
    ]

    model.to('mps')  # moving to mps for Mac (can alternatively do 'cpu')

    print()
    print(f"{'Untrained' if checkpoint == UNTRAINED_CHECKPOINT else 'Trained'} model predictions:")
    print("--------------------------")
    for text in text_list:
        # moving to mps for Mac (can alternatively do 'cpu')
        inputs = tokenizer.encode(text, return_tensors="pt").to("mps")

        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices

        print(text + " - " + id2label[predictions.tolist()[0]])


def main():
    untrained_model = load_model(checkpoint=UNTRAINED_CHECKPOINT)
    evaluate_model(untrained_model, checkpoint=UNTRAINED_CHECKPOINT)

    model = load_model()
    evaluate_model(model)


if __name__ == "__main__":
    main()
