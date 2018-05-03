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
    train(
        args.model_path,
        args.pre_trained_path,
        args.optimizer_path,
        args.corpus_path,
        args.caption_path,
        args.image_dir,
        args.log_step,
        args.save_step,
        args.num_epochs,
        args.batch_size,
        args.num_workers,
        args.lr,
        args.w_decay)


def train(
        num_epochs,
        pre_trained_path,
        optimizer_path,
        use_cuda=True,
        model_path='models/',
        corpus_path='data/corpus.pkl',
        caption_path='data/annotations/captions_train2017.json',
        image_dir='data/train',
        log_step=1,
        batch_size=200,
        num_workers=0,
        lr=0.0001,
        w_decay=0):
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # VGG feature extractor
    extractor = Vgg16Extractor(use_gpu=use_cuda, transform=False)

    # Load vocabulary wrapper.
    corpus = Corpus.load(FilePathManager.resolve(corpus_path))
    print(corpus.word_from_index(0))

    dataset = CocoDataset(corpus, root=image_dir, annFile=caption_path, transform=extractor.tf_image)
    # Build data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=use_cuda)

    # Build the models
    model = m_RNN()

    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    # criterion = nn.CrossEntropyLoss(ignore_index=corpus.word_index(corpus.PAD))
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=w_decay)

    # Continue Training
    if pre_trained_path is not None and optimizer_path is not None:
        model.load_state_dict(torch.load(pre_trained_path))
        optimizer.load_state_dict(torch.load(optimizer_path))

    # Train the Models
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (images, inputs, targets) in enumerate(dataloader):
            images = images.cuda()
            images_features, images_regions = extractor.forward(images)
            for k in range(inputs.shape[1]):
                # Set mini-batch dataset
                images = handle(images, cuda=use_cuda)
                input = handle(inputs[:, k, :-1], cuda=use_cuda)
                target = handle(targets[:, k, 1:], cuda=use_cuda)

                input = pack_padded_sequence(input, [17] * input.shape[0], True)[0]
                target = pack_padded_sequence(target, [17] * target.shape[0], True)[0]

                # Forward, Backward and Optimize
                model.zero_grad()

                # make update
                output = model(images_features, images_regions, input)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()
            # Print log info
            if i % log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, num_epochs, i, total_step,
                         loss.item(), np.exp(loss.item())))

        # Save the models
        torch.save(model.state_dict(),
                   os.path.join(model_path,
                                'model-%d.pkl' % (epoch + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(model_path,
                                'optimizer-%d.pkl' % (epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--pre_trained_path', type=str,
                        help='path for saved trained models')
    parser.add_argument('--optimizer_path', type=str,
                        help='path for saved  optimizer')
    parser.add_argument('--corpus_path', type=str, default='data/corpus.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--caption_path', type=str,
                        default='D:/Datasets/mscoco/test/captions_train2017.json',
                        help='path for train annotation json file')
    parser.add_argument('--image_dir', type=str, default='D:/Datasets/mscoco/test/images',
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int, default=1,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=2,
                        help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--w_decay', type=float, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
