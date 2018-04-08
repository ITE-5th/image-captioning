import torch
from torch import nn
from torch.autograd import Variable


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0, X_in.data))

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


if __name__ == '__main__':
    text = 'Hello World'
    text = [ord(c) for c in text]  # split and convert to ASCII
    text = Variable(torch.Tensor(text))
    one_hot = One_Hot(256)
    output = one_hot(text)
    print(output)
