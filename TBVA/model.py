import torch.nn as nn
from torchvision import models

from TBVA.multi_modal_layer import MultiModalLayer


class m_RNN(nn.Module):
    def __init__(self,
                 answer_length,
                 embed2_features,
                 rnn_features,
                 attention_features,
                 cnn_features,
                 L, D,
                 vocab_count=9956,
                 multimodal_features=2048,
                 ):
        super().__init__()
        self.answer_length = answer_length

        embeds_1_size = 1024
        embeds_2_size = 2048
        rnn_size = 2048
        attention_size = 1024
        cnn_features_size = 1024
        multimodal_in_size = 2048
        multimodal_out_size = 1024
        vgg16 = models.vgg16(pretrained=True)
        modules = list(vgg16.children())[:-1]  # delete the last fc layer.
        self.feature_extractor = nn.Sequential(*modules)
        self.attention_layer = nn.Linear(vgg16.fc.in_features, attention_size)

        self.embeds_1 = nn.Embedding(answer_length, embeds_1_size)
        self.embeds_2 = nn.Embedding(embeds_1_size, embeds_2_size)

        rnn_layers = 1
        self.rnn_cell = nn.LSTMCell(embeds_2_size, rnn_size, rnn_layers)

        self.multi_modal = MultiModalLayer(embeds_2_size, rnn_size, attention_size, cnn_features_size,
                                           multimodal_in_size, multimodal_out_size)

        # self.softmax = F.softmax(self.intermediate, vocab_count)

    def forward(self, image):
        return None


if __name__ == '__main__':
    print('M_RNN')
    x = m_RNN()
