import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models, transforms

from TBVA.multi_modal_layer import MultiModalLayer


class m_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        answer_length = 20
        embeds_1_size = 1024
        embeds_2_size = 2048
        rnn_size = 2048
        attention_size = 1024
        cnn_features_size = 1024
        multimodal_in_size = 2048
        multimodal_out_size = 1024
        rnn_layers = 1
        self.vocab_count = 9999
        self.vgg16 = models.vgg16(pretrained=True)

        for param in vgg16.parameters():
            param.requires_grad = False

        self.feature_extractor = nn.Linear(vgg16.fc.in_features, cnn_features_size)

        self.attention_layer = nn.Linear(cnn_features_size, attention_size)

        self.embeds_1 = nn.Embedding(answer_length, embeds_1_size)
        self.embeds_2 = nn.Embedding(embeds_1_size, embeds_2_size)

        self.rnn_cell = nn.LSTMCell(embeds_2_size, rnn_size, rnn_layers)

        self.multi_modal = MultiModalLayer(embeds_2_size, rnn_size, attention_size, cnn_features_size,
                                           multimodal_in_size, multimodal_out_size)
        self.intermediate = nn.Linear(multimodal_out_size, answer_length)
        # self.softmax = F.softmax(self.intermediate, vocab_count)
        self.init_weights()

    def init_weights(self):
        # cnn fc
        self.feature_extractor.weight.data.normal_(0.0, 0.02)
        self.feature_extractor.bias.data.fill_(0)

    def forward(self, image, captions, lengths):
        vgg_features = self.vgg16(image)
        img_features = self.feature_extractor(vgg_features)
        attn_features = self.attention_layer(img_features)

        embeddings = self.embeds_1(captions)
        embeddings_2 = self.embeds_2(embeddings)

        hiddens, _ = self.rnn(embeddings_2)

        mm_features = self.multi_modal(embeddings_2, hiddens[0], attn_features, img_features)
        intermediate_features = self.intermediate(mm_features)

        return F.softmax(intermediate_features, self.vocab_count)


imsize = 224

loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    _image = Image.open(image_name)
    _image = loader(_image).float()
    _image = Variable(_image, requires_grad=True)
    _image = _image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return _image.cuda()  # assumes that you're using GPU


if __name__ == '__main__':
    print('M_RNN')
    # x = m_RNN()
    # print(x.parameters())
    test_img = image_loader('../misc/images/1.jpg')
    vgg16 = models.vgg16(pretrained=True)
    vgg16.cuda()
    x = vgg16(test_img)
    print(x.shape)
    print('test')
