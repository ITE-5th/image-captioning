import pretrainedmodels
import torch
import torch.nn as nn
from pretrainedmodels import utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from misc.coco_dataset import CocoDataset
from misc.corpus import Corpus
from misc.file_path_manager import FilePathManager
from multi_modal_layer import MultiModalLayer


class m_RNN(nn.Module):
    def __init__(self, use_cuda=True):
        super().__init__()
        embeds_1_size = 1024
        embeds_2_size = 2048
        rnn_size = 512
        attention_size = 512
        cnn_features_size = 4096
        multimodal_in_size = 512
        multimodal_out_size = 1024
        rnn_layers = 1
        self.hidden_dim = 512
        self.vocab_count = 10496
        self.D = 49
        self.L = 512
        self.use_cuda = use_cuda

        # attention
        self.att_vw = nn.Linear(self.L, self.L, bias=False)
        self.att_hw = nn.Linear(rnn_size, self.L, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(self.D))
        self.att_w = nn.Linear(self.L, 1, bias=False)

        # TODO embedding change first linear to embedding and modify data loader
        self.embeds_1 = nn.Linear(self.vocab_count, embeds_1_size)
        self.embeds_2 = nn.Linear(embeds_1_size, embeds_2_size)

        self.rnn_cell = nn.LSTM(embeds_2_size, rnn_size, rnn_layers)

        self.multi_modal = MultiModalLayer(embeds_2_size, rnn_size, cnn_features_size,
                                           multimodal_in_size, multimodal_out_size)
        self.intermediate = nn.Linear(multimodal_out_size, self.vocab_count)

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
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context, alpha

    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = torch.zeros(1, batch_size, hidden_dim)
        c0 = torch.zeros(1, batch_size, hidden_dim)
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def get_image_features(self, img):
        feat_49 = self.vgg16_49(img)
        out_feat = feat_49.view(feat_49.size(0), -1)
        feat_4096 = self.vgg16_4096(out_feat)
        feat_49 = feat_49.view(feat_49.size(0), 49, feat_49.size(1))
        return feat_49, feat_4096

    def forward(self, image_features, image_regions, captions):

        images_count = image_features.shape[0]
        sentence_length = 17
        batch_size = images_count * sentence_length
        image_features = torch.stack([image_features] * sentence_length) \
            .permute(1, 0, 2) \
            .contiguous() \
            .view(-1, image_features.shape[-1])

        image_regions = torch.stack([image_regions.view(images_count, -1)] * sentence_length) \
            .permute(1, 0, 2) \
            .contiguous() \
            .view(-1, image_regions.shape[1], image_regions.shape[2])

        # image_regions = image_regions.repeat(sentence_length, 1, 1)
        # image_features = image_features.repeat(sentence_length, 1)
        h0, c0 = self.get_start_states(images_count)

        embeddings = self.embeds_1(captions)
        embeddings_2 = self.embeds_2(embeddings)

        hiddens, next_state = self.rnn_cell(embeddings_2.view(sentence_length, images_count, 2048),
                                            (h0[:batch_size, :], c0[:batch_size, :]))
        attention_layer = self._attention_layer

        atten_features, alpha = attention_layer(image_regions, hiddens.view(captions.shape[0], 512))

        mm_features = self.multi_modal(embeddings_2, hiddens.view(batch_size, -1), atten_features, image_features)
        intermediate_features = self.intermediate(mm_features)

        return nn.Softmax()(intermediate_features)

    def sample(self, image_features, image_regions, start_word):

        h0, c0 = self.get_start_states(1)

        word = start_word
        sampled_ids = []

        for step in range(17):
            embeddings = self.embeds_1(word)
            embeddings_2 = self.embeds_2(embeddings)

            hiddens, next_state = self.rnn_cell(embeddings_2.view(1, 1, 2048),
                                                (h0[:1, :], c0[:1, :]))
            attention_layer = self._attention_layer

            atten_features, alpha = attention_layer(image_regions, hiddens.view(1, 512))

            mm_features = self.multi_modal(embeddings_2, hiddens.view(1, -1), atten_features, image_features)
            intermediate_features = self.intermediate(mm_features)
            word = nn.Softmax()(intermediate_features)
            sampled_ids.append(word.squeeze().max(0)[1])

        return torch.stack(sampled_ids, 0)


def to_var(x, cuda=True):
    if cuda and torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


if __name__ == '__main__':
    use_cuda = True

    model = pretrainedmodels.vgg16()
    model.eval()
    tf_img = utils.TransformImage(model)

    corpus = Corpus.load(FilePathManager.resolve("../data/corpus.pkl"))
    dataset = CocoDataset(corpus, transform=tf_img)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=use_cuda)
    x = m_RNN()
    if use_cuda:
        x.cuda()
    for i, (images, captions, lengths) in enumerate(dataloader):
        for k in range(captions.shape[1]):
            test_img = to_var(images, cuda=use_cuda)
            captions_var = to_var(captions, cuda=use_cuda)
            inputs = captions_var[:, k, :-1]
            targets = captions_var[:, k:, 1:]
            inputs = pack_padded_sequence(inputs, [17] * inputs.shape[0], True)[0]
            print(k)

            start_word = to_var(corpus.word_one_hot('<start>'), cuda=use_cuda)
            sampled_ids = x.sample(test_img[0].unsqueeze(0), start_word)
            sampled_ids = sampled_ids.cpu().data.numpy()
            sentence = ''
            for i in sampled_ids:
                sentence += corpus.word_from_index(i) + ' '
            print(sentence)
