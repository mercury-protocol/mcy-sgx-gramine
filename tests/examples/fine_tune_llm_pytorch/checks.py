import evaluate
from tests.examples.fine_tune_llm_pytorch.user_script import *


if __name__ == "__main__":
    # {'accuracy': 0.533}
    from pprint import pprint

    data_loader = create_data_loader()
    model = create_model()
    optimizer = create_optimizer(model)
    extra_args = create_extra_training_args(data_loader, optimizer)

    print()
    print("training loop start")

    for epoch in range(N_EPOCHS):
        for batch in data_loader:
            train_batch(batch, model, optimizer, *extra_args)

    torch.save(model.state_dict(), "trained_reference_model.pth")

    print()
    print("training loop finish")

    eval_dataloader = create_eval_data_loader()
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    computed_metric = metric.compute()

    print()
    pprint(computed_metric)
