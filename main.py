import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import model

np.set_printoptions(suppress=True)


def get_FMNIST_data():
    # data reinforcement
    train_data = datasets.FashionMNIST('FMNIST/',
                                       train=True,
                                       download=True,
                                       transform=transforms.Compose([
                                           # transforms.RandomGrayscale(),
                                           # transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转，-10到10度之间随机选
                                           #
                                           # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                                           # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                           #
                                           # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),  # 随机视角
                                           # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 随机选择的高斯模糊模糊图像
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,)),
                                       ]))

    test_data = datasets.FashionMNIST('FMNIST/',
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          # transforms.RandomGrayscale(),

                                          # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                                          # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转

                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,)),
                                      ]))
    label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                  7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    return train_data, test_data, label_dict


def predict(model, images):
    model.cpu()
    with no_grad():
        model.eval()
        output = loaded_model.forward(test_images)
        test_loss = criterion(output, test_labels)
        result = torch.argmax(output, dim=1)
        result_labels = [label_dict[x] for x in result.numpy()]
    return (test_loss, result, result_labels)


def model_training(toTrain: bool):
    if not toTrain:
        return
    else:
        for epoch in range(1, epochs + 1):
            print("EPOCH:", epoch, end=" ")
            running_loss = 0
            running_acc = 0

            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()
                X = images.view(-1, 784)
                # X = images

                optimizer.zero_grad()
                output = model.forward(X)
                loss = criterion(output, labels)

                result = torch.argmax(output, dim=1)
                running_loss += loss.item()
                running_acc += torch.mean((result == labels).type(torch.float))

                loss.backward()
                optimizer.step()
            else:
                train_loss = running_loss / len(train_loader)
                train_acc = running_acc / len(train_loader)

                with no_grad():
                    model.eval()
                    output = model.forward(val_images)
                    val_loss = criterion(output, val_labels)
                    model.train()

                result_labels = torch.argmax(output, dim=1)
                accuracy = torch.mean((result_labels == val_labels).type(torch.float))

                train_loss_pe.append(train_loss)
                val_loss_pe.append(val_loss.item())

                train_acc_pe.append(train_acc.item())
                val_acc_pe.append(accuracy.item())

                print("Training Loss: {:.3f}".format(train_loss), end=" ")
                print("Val Loss: {:.3f}".format(val_loss), end=" ")
                print("Train Accuracy: {:.2f}%".format(train_acc.item() * 100), end=" ")
                print("Val Accuracy: {:.2f}%".format(accuracy.item() * 100), end=" \n")

                with open("Models/mlp_fmnist_model_{}.pth".format(epoch), "wb") as f:
                    model.eval()
                    pickle.dump(model, f)
                    model.train()

        # visualise training loss
        plots = [(train_loss_pe, val_loss_pe), (train_acc_pe, val_acc_pe)]
        plt_labels = [("Training Loss", "Validation Loss"), ("Training Accuracy", "Validation Accuracy")]
        plt_titles = ["Loss", "Accuracy"]
        plt.figure(figsize=(20, 7))
        for i in range(0, 2):
            ax = plt.subplot(1, 2, i + 1)
            ax.plot(plots[i][0], label=plt_labels[i][0])
            ax.plot(plots[i][1], label=plt_labels[i][1])
            ax.set_title(plt_titles[i])
            ax.legend()


if __name__ == '__main__':
    # hyperparameters
    epochs = 200
    learning_rate = 0.0007
    batch_size = 64

    toTrain = True
    toModel = model.NNewMLPNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get FMNIST data and label
    train_data, test_data, label_dict = get_FMNIST_data()

    # generate iterators
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=5000)

    # handle val and test data
    val_test_data = list(iter(test_loader))
    val_images, val_labels = val_test_data[0]
    test_images, test_labels = val_test_data[1]

    # initialise the model
    model = toModel

    # Input size
    input_size = (batch_size, 784)
    input = torch.randn(*input_size)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(input)

    # loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if device.type == 'cuda':
        model.to(device)
        val_images, val_labels = val_images.cuda(), val_labels.cuda()

    # train model
    train_loss_pe = list()
    val_loss_pe = list()
    train_acc_pe = list()
    val_acc_pe = list()
    model_training(toTrain)

    # visiualise
    plots = [(train_loss_pe, val_loss_pe), (train_acc_pe, val_acc_pe)]
    plt_labels = [("Training Loss", "Validation Loss"), ("Training Accuracy", "Validation Accuracy")]
    plt_titles = ["Loss", "Accuracy"]
    plt.figure(figsize=(20, 7))
    for i in range(0, 2):
        ax = plt.subplot(1, 2, i + 1)
        ax.plot(plots[i][0], label=plt_labels[i][0])
        ax.plot(plots[i][1], label=plt_labels[i][1])
        ax.set_title(plt_titles[i])
        ax.legend()
    plt.show()

    # load latest model and predict
    with open("Models/mlp_fmnist_model_" + str(epochs) + ".pth", "rb") as f:
        loaded_model = pickle.load(f)
    test_loss, result, result_labels = predict(loaded_model, test_images)

    accuracy = torch.mean((result == test_labels).type(torch.float))
    print("Test Loss: {:.2f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

    print(1)
