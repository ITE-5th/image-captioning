import argparse
import glob

import torch
from PIL import Image
from pretrainedmodels import utils

from file_path_manager import FilePathManager
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
    extractor = Vgg16Extractor(use_gpu=use_cuda, transform=True, regions_count=args.image_regions)
    load_img = utils.LoadImage()

    corpus = Corpus.load(FilePathManager.resolve(args.corpus_path))
    model = m_RNN(use_cuda=use_cuda, image_regions=args.image_regions)
    start_word = torch.LongTensor([corpus.word_index('<start>')])

    if use_cuda:
        model.cuda()
        start_word = start_word.cuda()

    print('loading model')
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    captions = []
    for j in range(len(image_list)):
        image_path = image_list[j]
        features, regions = extractor.forward(load_img(image_path))
        # sampled_ids = model.sample(features, regions, start_word.unsqueeze(0))
        sampled_ids, alphas = model.sample(features, regions, start_word)

        alphas = torch.cat(alphas[1:], 0)
        sampled_ids = sampled_ids.cpu().data.numpy()
        sentence = ''
        words = []
        for j in sampled_ids:
            word = corpus.word_from_index(j)
            words.append(word)
            sentence += word + ' '

        print(f'image {image_path} : {sentence}')
        captions.append(sentence)
        attention_visualization(image_path, words, alphas.data.cpu(), args.image_regions)

    return captions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='misc\\images',
                        help='input image for generating caption')
    parser.add_argument('--corpus_path', type=str, default='data\\corpus.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--model_path', type=str, default='models\\49\\model-14.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_regions', type=int, default=49,
                        help='number of image regions to be extracted (49 or 196)')

    args = parser.parse_args()
    main(args)
