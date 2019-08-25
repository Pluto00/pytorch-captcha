import torch.nn as nn
import config


# CNN Model (4 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 1 * 160 * 60
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(  # 32 * 80 * 30
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(  # 64 * 40 * 15
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            # nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(  # 512 * 20 * 7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(  # 512 * 10 * 3
            nn.Linear((config.IMAGE_WIDTH // 16) * (config.IMAGE_HEIGHT // 16) * 512,
                      config.CAPTCHA_LEN * config.CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)  # 把out拉直，reshape
        out = self.fc(out)
        return out
