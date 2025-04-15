from neural_network import *
from torch import optim
from dataset import *
from torcheval.metrics import BinaryAccuracy

X_train, y_train, X_test, y_test = test_train_split()
batches = list(torch.split(X_train, 1))
batches_of_labels = list(torch.split(y_train, 1))

def unsup_training():

    unsup_model = AutoEncoder()
    loss_fn_unsup = nn.MSELoss()

    optimizer = optim.Adam(unsup_model.parameters(), lr=0.01, weight_decay=0.001)
    unsup_model.train()

    print("\nUnsupervised part!")
    for epoch in range(1):
        for i in range(len(batches)):
            optimizer.zero_grad()

            outputs = unsup_model(batches[i].reshape(1, 1, 513, 513))


            loss = loss_fn_unsup(outputs, batches[i].reshape(1, 1, 513, 513))
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")
    return unsup_model


def sup_training(model):
    sup_model = LastLayer(model)

    sup_model.train()

    loss_fn_sup = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    print("\nSupervised part!")
    for epoch in range(1):
        for i in range(len(batches)):
            optimizer.zero_grad()
            outputs = (sup_model(batches[i].reshape(1, 1, 513, 513)))
            print(outputs, batches_of_labels[i])
            loss = loss_fn_sup(outputs, batches_of_labels[i])
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return sup_model

def acc(model):
    pred_labels = model(X_test.reshape(80, 1, 513, 513)).argmax(dim=1)

    acc = (pred_labels == y_test).float().mean().item()

    print("\nAccuracy = ")
    print(acc * 100)
    return acc

if __name__ == "__main__":
    unsup_model = unsup_training()
    sup_model = sup_training(unsup_model)
    acc(sup_model)