import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from file_path_manager import FilePathManager
from misc.coco_dataset import CocoDataset
from misc.corpus import Corpus
from model import m_RNN
from vgg16_extractor import Vgg16Extractor


def handle(x, cuda):
    if cuda and torch.cuda.is_available():
        x = x.cuda()
    return x


def main(args):
    use_cuda = True
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # VGG feature extractor
    extractor = Vgg16Extractor(use_gpu=use_cuda, transform=False)

    # Load vocabulary wrapper.
    corpus = Corpus.load(FilePathManager.resolve(args.corpus_path))
    print(corpus.word_from_index(0))

    dataset = CocoDataset(corpus, transform=extractor.tf_image)
    # Build data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=use_cuda)

    # Build the models
    model = m_RNN()

    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(dataloader)
    for epoch in range(args.num_epochs):
        for i, (images, inputs, targets) in enumerate(dataloader):
            images = images.cuda()
            images_features, images_regions = extractor.forward(images)
            for k in range(inputs.shape[1]):
                # Set mini-batch dataset
                images = handle(images, cuda=use_cuda)
                input = handle(inputs[:, k, :-1, :], cuda=use_cuda)
                target = handle(targets[:, k, 1:], cuda=use_cuda)

                input = pack_padded_sequence(input, [17] * input.shape[0], True)[0]
                target = pack_padded_sequence(target, [17] * target.shape[0], True)[0]

                # Forward, Backward and Optimize
                model.zero_grad()

                output = model(images_features, images_regions, input)

                loss = criterion(output, target)
                # TODO TWS
                loss.backward()
                optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))

                # Save the models
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.model_path,
                                        'model-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--corpus_path', type=str, default='data/corpus.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='D:/Datasets/mscoco/2014/resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)