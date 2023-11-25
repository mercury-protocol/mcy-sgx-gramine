from pytorch.normal.eval import evaluate_training, make_predictions

from user_script import load_network, train, test, test_loader, train_losses, train_counter, test_losses, test_counter


N_EPOCHS = 2

# ------------------- train the model ----------------

test(0)
for e in range(1, N_EPOCHS + 1):
    train(e)
    test(e)

# ------------------- evaluate the network ----------------
network = load_network()
evaluate_training(train_counter, train_losses, test_counter, test_losses)
make_predictions(network, test_loader)
