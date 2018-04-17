import pickle

import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pretrainedmodels import utils
from torch.autograd import Variable
from torchvision import models, transforms

from TBVA.multi_modal_layer import MultiModalLayer
from misc.build_vocab import Vocabulary
from misc.data_loader import get_loader


class m_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        answer_length = 20
        embeds_1_size = 1024
        embeds_2_size = 2048
        rnn_size = 512
        attention_size = 512
        cnn_features_size = 4096
        multimodal_in_size = 512
        multimodal_out_size = 1024
        rnn_layers = 1
        self.vocab_count = 9999
        self.D = 49
        self.L = 512

        vgg16 = models.vgg16(pretrained=True)
        self.vgg16_49 = nn.Sequential(*list(vgg16.children())[:-1][0])
        self.vgg16_4096 = nn.Sequential(*list(list(vgg16.children())[-1])[:-3])

        for param in vgg16.parameters():
            param.requires_grad = False

        # attention
        self.att_vw = nn.Linear(self.L, self.L, bias=False)
        self.att_hw = nn.Linear(rnn_size, self.L, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(self.D))
        self.att_w = nn.Linear(self.L, 1, bias=False)

        self.embeds_1 = nn.Embedding(self.vocab_count, embeds_1_size)
        self.embeds_2 = nn.Embedding(embeds_1_size, embeds_2_size)

        self.rnn_cell = nn.LSTMCell(embeds_2_size, rnn_size, rnn_layers)

        self.multi_modal = MultiModalLayer(embeds_2_size, rnn_size, attention_size, cnn_features_size,
                                           multimodal_in_size, multimodal_out_size)
        self.intermediate = nn.Linear(multimodal_out_size, answer_length)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * 49 * 512
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea * att_h * self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full).squeeze(2)
        alpha = nn.Softmax()(att_out)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2))
        return context, alpha

    def init_weights(self):
        # cnn fc
        self.feature_extractor.weight.data.normal_(0.0, 0.02)
        self.feature_extractor.bias.data.fill_(0)

    def get_image_features(self, img):
        feat_49 = self.vgg16_49(img)
        out_feat = feat_49.view(feat_49.size(0), -1)
        feat_4096 = self.vgg16_4096(out_feat)
        feat_49 = feat_49.view(feat_49.size(0), 49, feat_49.size(1))
        return feat_49, feat_4096

    def forward(self, image, captions):
        vgg_features, image_attention_features = self.get_image_features(image)

        embeddings = self.embeds_1(captions)
        embeddings_2 = self.embeds_2(embeddings)
        hiddens, next_state = self.rnn(embeddings_2)
        attention_layer = self._attention_layer

        atten_features, alpha = attention_layer(image_attention_features, hiddens)

        mm_features = self.multi_modal(embeddings_2, hiddens, atten_features, vgg_features)
        intermediate_features = self.intermediate(mm_features)

        return F.softmax(intermediate_features, self.vocab_count)


imsize = 224

loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])


def image_loader(image_name, cuda=True):
    """load image, returns cuda tensor"""
    _image = Image.open(image_name)
    _image = loader(_image).float()
    _image = Variable(_image, requires_grad=True)
    _image = _image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return _image.cuda() if cuda else _image


model = pretrainedmodels.vgg16()
model.eval()
tf_img = utils.TransformImage(model)
Vocabulary()
with open('..\\data\\vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

data_loader = get_loader('D:\\Datasets\\mscoco\\2014\\train',
                         'D:\\Datasets\\mscoco\\2014\\annotations_trainval2014\\captions_train2014.json',
                         vocab,
                         tf_img, 4,
                         shuffle=True, num_workers=0)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


if __name__ == '__main__':
    # test_img = image_loader('../misc/images/1.jpg')
    for i, (images, captions, lengths) in enumerate(data_loader):
        test_img = to_var(images, volatile=True)
        break

    print('M_RNN')
    x = m_RNN()
    x.cuda()
    a, b = x.get_image_features(test_img)
    # print(x.parameters())
    print('test')
