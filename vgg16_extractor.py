import pretrainedmodels
import torch.nn as nn
from pretrainedmodels import utils

from file_path_manager import FilePathManager


class Vgg16Extractor:

    def __init__(self, use_gpu: bool = True, transform: bool = True):
        super().__init__()

        self._cnn = pretrainedmodels.vgg16()
        self.regions = nn.Sequential(*(self.cnn._features[:-2]))
        self.regions_out = nn.Sequential(*(self.cnn._features[-2:]))

        self.tf_image = utils.TransformImage(self.cnn)
        self.transform = transform
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.cnn = self.cnn.cuda()
        self.cnn.eval()

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, image):
        if self.transform:
            image = self.tf_image(image)

        if len(image.size()) == 3:
            image = image.unsqueeze(0)

        if self.use_gpu:
            image = image.cuda()

        regions = self.regions(image)
        feat = self.regions_out(regions)
        x = feat.view(feat.size(0), -1)
        x = self.cnn.linear0(x)
        x = self.cnn.relu0(x)
        x = self.cnn.dropout0(x)
        features = self.cnn.linear1(x)
        return features, regions.view(regions.size(0), 196, regions.size(1))

    def __call__(self, image):
        return self.forward(image)


if __name__ == '__main__':
    extractor = Vgg16Extractor()
    load_img = utils.LoadImage()
    image_path = FilePathManager.resolve("misc/images/1.jpg")

    extractor.forward(load_img(image_path))
