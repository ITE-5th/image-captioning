import torch.nn.functional as F
from torch.nn import Module, Linear


class MultiModalLayer(Module):
    def __init__(self,
                 embed2_features,
                 rnn_features,
                 cnn_features,
                 multimodal_features,
                 out_features):
        super().__init__()

        # MultiModal Layers

        self.linear_embed2 = Linear(embed2_features, multimodal_features)
        self.linear_rnn = Linear(rnn_features, multimodal_features)
        self.linear_cnn = Linear(cnn_features, multimodal_features)
        self.fusion = Linear(multimodal_features, out_features)

    def forward(self, embed2, rnn, attention, cnn):
        w = self.linear_embed2(embed2)
        r = self.linear_rnn(rnn)
        v = self.linear_cnn(cnn)
        fusion = self.fusion(w + r + v + attention)

        return 1.7159 * F.tanh((2 / 3) * fusion)
