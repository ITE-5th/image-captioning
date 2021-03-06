import pretrainedmodels
import torch.nn as nn
from pretrainedmodels import utils

from file_path_manager import FilePathManager


class Vgg16Extractor:

    def __init__(self, regions_count=49, use_gpu: bool = True, transform: bool = True):
        super().__init__()
        print('USING VGG16')

        self.cnn = pretrainedmodels.vgg16()
        self.regions_count = regions_count
        self.regions_features_size = 512
        self.features_size = 4096

        if regions_count == 49:
            self.regions = self.cnn._features
        else:
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
        feat = regions
        if self.regions_count == 196:
            feat = self.regions_out(regions)
        x = feat.view(feat.size(0), -1)
        x = self.cnn.linear0(x)
        x = self.cnn.relu0(x)
        x = self.cnn.dropout0(x)
        features = self.cnn.linear1(x)
        return features, regions.view(regions.size(0), self.regions_count, regions.size(1))

    def __call__(self, image):
        return self.forward(image)


if __name__ == '__main__':
    extractor = Vgg16Extractor()
    load_img = utils.LoadImage()
    image_path = FilePathManager.resolve("misc/images/airplane.jpg")

    feat, reg = extractor.forward(load_img(image_path))
    print(feat.shape)
    print(reg.shape)
