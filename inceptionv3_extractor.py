import pretrainedmodels
import torch.nn.functional as F
from pretrainedmodels import utils

from file_path_manager import FilePathManager


class InceptionV3Extractor:

    def __init__(self, use_gpu: bool = True, transform: bool = True):
        super().__init__()
        print('USING InceptionV3Extractor')
        self.cnn = pretrainedmodels.inceptionv3()

        self.tf_image = utils.TransformImage(self.cnn)
        self.transform = transform
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.cnn = self.cnn.cuda()
        self.cnn.eval()
        self.features_size = 2048
        self.regions_count = 64
        self.regions_features_size = 2048

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, image):
        if self.transform:
            image = self.tf_image(image)

        if len(image.size()) == 3:
            image = image.unsqueeze(0)

        if self.use_gpu:
            image = image.cuda()

        regions = self.cnn.features(image)

        x = F.avg_pool2d(regions, kernel_size=8)  # 1 x 1 x 2048
        x = F.dropout(x)  # 1 x 1 x 2048
        features = x.view(x.size(0), -1)  # 2048

        return features, regions.view(regions.size(0), self.regions_count, regions.size(1))

    def __call__(self, image):
        return self.forward(image)


if __name__ == '__main__':
    load_img = utils.LoadImage()
    image_path = FilePathManager.resolve("misc/images/airplane.jpg")

    extractor = InceptionV3Extractor()
    feat, reg = extractor.forward(load_img(image_path))
    print(feat.shape)
    print(reg.shape)
