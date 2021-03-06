import argparse
import glob

import torch
from PIL import Image
from pretrainedmodels import utils

from file_path_manager import FilePathManager
from inceptionv3_extractor import InceptionV3Extractor
from misc.corpus import Corpus
from misc.helper import attention_visualization
from model import m_RNN
from vgg16_extractor import Vgg16Extractor


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def main(args):
    image_list = []
    for filename in glob.glob(f'{args.images_dir}\\*.jpg'):
        image_list.append(filename)
    if len(image_list) == 0:
        print(f'Sorry No .jpg images found in {args.images_dir}')
        return

    use_cuda = False

    if args.image_regions == 64:
        # Inception V3 feature extractor
        extractor = InceptionV3Extractor(use_gpu=use_cuda)
    else:
        # VGG feature extractor
        extractor = Vgg16Extractor(use_gpu=use_cuda, regions_count=args.image_regions)

    load_img = utils.LoadImage()

    corpus = Corpus.load(FilePathManager.resolve(args.corpus_path))

    # Build the models
    model = m_RNN(use_cuda=use_cuda,
                  image_regions=extractor.regions_count,
                  regions_features=extractor.regions_features_size,
                  features_size=extractor.features_size)

    start_word = torch.LongTensor([corpus.word_index('<start>')])

    if use_cuda:
        model.cuda()
        start_word = start_word.cuda()

    print('loading model')
    state_dict = torch.load(args.model_path, map_location=None if use_cuda else 'cpu')
    model.load_state_dict(state_dict)
    captions = []
    for i in range(len(image_list)):
        image_path = image_list[i]
        features, regions = extractor.forward(load_img(image_path))
        # sampled_ids = model.sample(features, regions, start_word.unsqueeze(0))
        sampled_ids, alphas = model.sample(features, regions, start_word, args.beam_size)

        sampled_ids = sampled_ids.cpu().data.numpy().T
        sentence = ''
        words = []
        for j in sampled_ids[0]:
            word = corpus.word_from_index(j)

            if len(word) == 0:
                continue
            words.append(word)

        for k in range(len(words)):
            sentence += (' ' if k != 0 and words[k] != ',' else '') + words[k]
        sentence = sentence.split("<end>")[0]
        print(f'image {image_path.replace(args.images_dir,"")} : {sentence}')
        captions.append(sentence)
        attention_visualization(image_path, words, alphas, args.image_regions)

    return captions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='misc\\images',
                        help='input image for generating caption')
    parser.add_argument('--corpus_path', type=str, default='data\\corpus.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--model_path', type=str, default='models\\49\\model-39.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_regions', type=int, default=49,
                        help='number of image regions to be extracted (49 or 196) 64 for inception_v3')
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()
    main(args)
