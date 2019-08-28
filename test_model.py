import numpy as np
import torch
from torch.autograd import Variable
import config, dataset, one_hot
from cnn_model import CNN


def main():
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.load_state_dict(torch.load('model.pkl'))
    cnn.eval()
    print("load cnn net.")

    test_dataloader = dataset.get_test_data_loader()

    correct = 0
    total = 0
    for i, (image, label) in enumerate(test_dataloader):
        vimage = Variable(image)
        if torch.cuda.is_available():
            vimage = vimage.cuda()
        output = cnn(vimage)
        predict_label = ""
        for j in range(4):
            predict_label += config.CHAR_SET[
                np.argmax(output[0, j * config.CHAR_SET_LEN:(j + 1) * config.CHAR_SET_LEN].data.cpu().numpy())
            ]
        true_label = one_hot.vec2text(label.numpy()[0])
        total += label.size(0)

        if predict_label == true_label:
            correct += 1
        if total % 100 == 0:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    main()

