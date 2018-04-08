import torch.nn as nn
from torchvision import models


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = models.inception_v3(pretrained=True)
        self.embeds_1 = nn.Embedding(2, 5)
        self.embeds_2 = nn.Embedding(2, 5)
        self.soft_max = nn.Softmax()

    def forward(self, image):
        return None


if __name__ == '__main__':
    Network()
    print('begin training')
