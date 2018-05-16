import torch
import torch.nn as nn
import torch.nn.functional as F

from beam_search import BeamSearch
from multi_modal_layer import MultiModalLayer


class m_RNN(nn.Module):
    def __init__(self, use_cuda=True, image_regions=49, regions_features=512, features_size=4096):
        super().__init__()

        embeds_1_size = 1024
        embeds_2_size = 2048
        rnn_size = 512
        cnn_features_size = features_size
        multimodal_out_size = 1024
        rnn_layers = 1
        self.hidden_dim = 512
        self.vocab_count = 10496
        self.D = image_regions
        self.L = regions_features
        self.use_cuda = use_cuda

        # attention
        self.att_vw = nn.Linear(self.L, self.L, bias=False)
        self.att_hw = nn.Linear(rnn_size, self.L, bias=False)
        self.att_bias = nn.Parameter(torch.ones(self.D))
        self.att_w = nn.Linear(self.L, 1, bias=False)

        self.embeds_1 = nn.Embedding(self.vocab_count, embeds_1_size)

        self.embeds_2 = nn.Linear(embeds_1_size, embeds_2_size)

        self.rnn_cell = nn.LSTM(embeds_2_size, rnn_size, rnn_layers)

        self.multi_modal = MultiModalLayer(embeds_2_size, rnn_size, cnn_features_size,
                                           self.L, multimodal_out_size)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * D * L
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea * att_h)
        att_out = self.att_w(att_full).squeeze(2)
        alpha = F.softmax(att_out, dim=1)
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

    def forward(self, image_features, image_regions, captions):

        images_count = image_features.shape[0]
        sentence_length = 17
        batch_size = images_count * sentence_length
        # image_features = torch.stack([image_features] * sentence_length) \
        #     .permute(1, 0, 2) \
        #     .contiguous() \
        #     .view(-1, image_features.shape[-1])
        #
        # image_regions = torch.stack([image_regions.view(images_count, -1)] * sentence_length) \
        #     .permute(1, 0, 2) \
        #     .contiguous() \
        #     .view(-1, image_regions.shape[1], image_regions.shape[2])

        image_regions = image_regions.repeat(sentence_length, 1, 1)
        image_features = image_features.repeat(sentence_length, 1)
        h0, c0 = self.get_start_states(images_count)

        embeddings = self.embeds_1(captions)
        embeddings_2 = self.embeds_2(embeddings)

        hiddens, next_state = self.rnn_cell(embeddings_2.view(sentence_length, images_count, 2048),
                                            (h0[:batch_size, :], c0[:batch_size, :]))
        attention_layer = self._attention_layer

        atten_features, alpha = attention_layer(image_regions, hiddens.view(captions.shape[0], 512))

        mm_features = self.multi_modal(embeddings_2, hiddens.view(batch_size, -1), atten_features, image_features)
        # intermediate_features = self.intermediate(mm_features)
        intermediate_features = F.linear(mm_features, weight=self.embeds_1.weight)

        # return nn.Softmax()(intermediate_features)
        return intermediate_features

    def sample(self, image_features, image_regions, start_word, beam_size=5):
        images_count = image_features.shape[0]
        sentence_length = 17
        batch_size = images_count * beam_size
        h0, c0 = self.get_start_states(batch_size)
        # image_regions = image_regions.repeat(beam_size, 1, 1)
        # image_features = image_features.repeat(beam_size, 1)
        image_features = torch.stack([image_features] * beam_size) \
            .permute(1, 0, 2) \
            .contiguous() \
            .view(-1, image_features.shape[-1])

        image_regions = torch.stack([image_regions.view(images_count, -1)] * beam_size) \
            .permute(1, 0, 2) \
            .contiguous() \
            .view(-1, image_regions.shape[1], image_regions.shape[2])

        word = start_word.repeat(batch_size)
        alphas = []
        all_words_indices = []
        beam_searcher = BeamSearch(beam_size, images_count, sentence_length)
        for step in range(17):
            if self.use_cuda:
                word = word.cuda()
            embeddings = self.embeds_1(word)
            embeddings_2 = self.embeds_2(embeddings)

            hiddens, (h0, c0) = self.rnn_cell(embeddings_2.view(1, batch_size, 2048),
                                              (h0, c0))
            attention_layer = self._attention_layer

            atten_features, alpha = attention_layer(image_regions, hiddens.view(batch_size, 512))
            # images count * beam size * regions
            alphas.append(alpha.reshape(images_count, beam_size, -1))

            mm_features = self.multi_modal(embeddings_2, hiddens.view(batch_size, -1), atten_features, image_features)
            # intermediate_features = self.intermediate(mm_features)
            intermediate_features = F.linear(mm_features, weight=self.embeds_1.weight)
            beam_indices, words_indices = beam_searcher.expand_beam(outputs=intermediate_features)
            words_indices = torch.tensor(words_indices)
            # images count * beam size * word index
            all_words_indices.append(words_indices.reshape(images_count, beam_size))
            word = words_indices

        results = beam_searcher.get_results()
        if images_count == 1:
            for j in range(images_count):
                for i in range(len(results)):
                    nonzero = (all_words_indices[i][j] == results[i][j]).nonzero()[0]
                    alphas[i] = alphas[i][j][nonzero].squeeze()
        else:
            alphas = []

        return results, alphas


if __name__ == '__main__':
    model = m_RNN()

    model.train()
    model.cuda()

    for name, param in model.named_parameters():
        print(name, param.size())
