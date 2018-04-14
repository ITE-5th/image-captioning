import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torchvision import models


class m_RNN(nn.Module):
    def __init__(self,
                 answer_length,
                 vocab_count,
                 embed2_features,
                 rnn_features,
                 attention_features,
                 cnn_features,
                 L, D,
                 multimodal_features=2048,
                 ):
        super().__init__()
        self.answer_length = answer_length

        vgg16 = models.vgg16(pretrained=True)
        modules = list(vgg16.children())[:-1]  # delete the last fc layer.
        self.feature_extractor = nn.Sequential(*modules)
        self.attention_layer = nn.Linear(vgg16.fc.in_features, 1)

        self.embeds_1 = nn.Embedding(answer_length, 1024)
        self.embeds_2 = nn.Embedding(2, 2048)
        self.softmax = F.softmax(1024, vocab_count)

        # MultiModal Layers
        self.linear_embed2 = Linear(embed2_features, multimodal_features)
        self.linear_rnn = Linear(rnn_features, multimodal_features)
        self.linear_attention = Linear(attention_features, multimodal_features)
        self.linear_cnn = Linear(cnn_features, multimodal_features)

        self.intermediate = nn.Linear(multimodal_features, 1024)

    @staticmethod
    def g(x):
        return 1.7159 * numpy.tanh((2 / 3) * x)

    def forward(self, image):
        return None


if __name__ == '__main__':
    print('M_RNN')
    # m_RNN()
