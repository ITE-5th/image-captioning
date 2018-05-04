import argparse

import matplotlib.pyplot as plt
import numpy as np
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
    use_cuda = True
    extractor = Vgg16Extractor(use_gpu=use_cuda, transform=True)
    load_img = utils.LoadImage()

    corpus = Corpus.load(FilePathManager.resolve(args.corpus_path))
    model = m_RNN(use_cuda=use_cuda)

    start_word = torch.Tensor([corpus.word_index('<start>')])

    if use_cuda:
        model.cuda()
        start_word = start_word.cuda()

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    features, regions = extractor.forward(load_img(args.image))
    # sampled_ids = model.sample(features, regions, start_word.unsqueeze(0))
    sampled_ids, alphas = model.sample(features, regions, start_word)
    alphas = torch.cat(alphas[1:], 0)
    sampled_ids = sampled_ids.cpu().data.numpy()
    sentence = ''
    words = []
    for i in sampled_ids:
        words.append(corpus.word_from_index(i))
        sentence += corpus.word_from_index(i) + ' '

    attention_visualization(args.image, words, alphas.data.cpu())
    print(sentence)

    image = Image.open(args.image)
    plt.imshow(np.asarray(image))

    return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='misc/images/Image.jpg',
                        help='input image for generating caption')
    parser.add_argument('--corpus_path', type=str, default='data/corpus.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--model_path', type=str, default='models/model-10.pkl',
                        help='path for vocabulary wrapper')

    args = parser.parse_args()
    main(args)
