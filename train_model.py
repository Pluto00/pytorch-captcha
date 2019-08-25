import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import dataset
from cnn_model import CNN

# Hyper Parameters
num_epochs = 50
learning_rate = 0.001


def main():
    cnn = CNN()
    loss_func = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        cnn.cuda()
        loss_func.cuda()

    # Train the Model
    train_dataloader = dataset.get_train_data_loader()
    cnn.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.float())
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            predict_labels = cnn(images)
            loss = loss_func(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
        if (epoch + 1) % 10 == 0:
            torch.save(cnn.state_dict(), "./model.pkl")
            print("save model...")
    torch.save(cnn.state_dict(), "./model.pkl")  # current is model1.pkl
    print("save last model")


if __name__ == '__main__':
    main()
