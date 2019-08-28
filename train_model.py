import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import dataset
from cnn_model import CNN
import one_hot
import config
import numpy as np

# Hyper Parameters
num_epochs = 50
learning_rate = 0.001


def main():
    # Load net
    cnn = CNN()
    loss_func = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        cnn.cuda()
        loss_func.cuda()

    # Load data
    train_dataloader = dataset.get_train_data_loader()
    test_dataloader = dataset.get_test_data_loader()

    # Train model
    for epoch in range(num_epochs):
        cnn.train()
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.long())
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

        # Save and test model
        if (epoch + 1) % 10 == 0:
            filename = "model" + str(epoch + 1) + ".pkl"
            torch.save(cnn.state_dict(), filename)
            cnn.eval()
            correct = 0
            total = 0
            for (image, label) in test_dataloader:
                vimage = Variable(image)
                if torch.cuda.is_available():
                    vimage = vimage.cuda()
                output = cnn(vimage)
                predict_label = ""
                for k in range(4):
                    predict_label += config.CHAR_SET[
                        np.argmax(output[0, k * config.CHAR_SET_LEN:(k + 1) * config.CHAR_SET_LEN].data.cpu().numpy())
                    ]
                true_label = one_hot.vec2text(label.numpy()[0])
                total += label.size(0)
                if predict_label == true_label:
                    correct += 1
                if total % 200 == 0:
                    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
            print("save and test model...")
    torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
    print("save last model")


if __name__ == '__main__':
    main()
