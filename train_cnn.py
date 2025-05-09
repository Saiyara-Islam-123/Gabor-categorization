import neural_network
from neural_network import *
from torch import optim
from dataset import *


X_train, y_train, X_test, y_test = test_train_split()
batches = list(torch.split(X_train, 1))
batches_of_labels = list(torch.split(y_train, 1))

def unsup_training():

    unsup_model = AutoEncoder()
    loss_fn_unsup = nn.MSELoss()

    optimizer = optim.Adam(unsup_model.parameters(), lr=0.01, weight_decay=0.01)
    unsup_model.train()

    print("\nUnsupervised part!")
    for epoch in range(1):
        for i in range(len(batches)):
            optimizer.zero_grad()

            outputs = unsup_model(batches[i].reshape(1, 1, 513, 513))


            loss = loss_fn_unsup(outputs, batches[i].reshape(1, 1, 513, 513))
            loss.backward()
            optimizer.step()

            print(loss.item())

        print(f"Loss: {loss.item()}")
    return unsup_model


def sup_training(unsup_model):
    sup_model = LastLayer(unsup_model)

    sup_model.train()

    loss_fn_sup = nn.CrossEntropyLoss()
    optimizer = optim.Adam(sup_model.parameters(), lr=0.005, weight_decay=0.0)

    print("\nSupervised part!")
    for epoch in range(1):
        for i in range(len(batches)):
            optimizer.zero_grad()
            outputs = (sup_model(batches[i].reshape(1, 1, 513, 513)))

            loss = loss_fn_sup(outputs, batches_of_labels[i])
            print(outputs, batches_of_labels[i], loss.item(), i, epoch)
            loss.backward()
            optimizer.step()

    return sup_model

def acc(model):
    pred_labels = model(X_test.reshape(80, 1, 513, 513)).argmax(dim=1)


    acc = (pred_labels == y_test).float().mean().item()
    print(pred_labels)
    print(y_test)

    print("\nAccuracy = ")
    print(acc * 100)
    return acc

if __name__ == "__main__":
    unsup_model = unsup_training()
    torch.save(unsup_model.state_dict(), "unsup_model.pth")

    unsup_model = neural_network.AutoEncoder()
    unsup_model.load_state_dict(torch.load("unsup_model.pth", weights_only=True), strict=True)

    sup_model = sup_training(unsup_model)
    acc(sup_model)