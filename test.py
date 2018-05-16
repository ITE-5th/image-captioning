import argparse
import json
import os
import os.path as osp

import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from file_path_manager import FilePathManager
from inceptionv3_extractor import InceptionV3Extractor
from misc.coco_dataset import CocoDataset
from misc.corpus import Corpus
from model import m_RNN
from pycocoevalcap.eval import COCOEvalCap
from vgg16_extractor import Vgg16Extractor


def handle(x, cuda):
    if cuda and torch.cuda.is_available():
        x = x.cuda()
    return x


def main(args):
    use_cuda = True

    if args.image_regions == 64:
        # Inception V3 feature extractor
        extractor = InceptionV3Extractor(use_gpu=use_cuda, transform=False)
    else:
        # VGG feature extractor
        extractor = Vgg16Extractor(use_gpu=use_cuda, transform=False, regions_count=args.image_regions)

    # Load vocabulary wrapper.
    corpus = Corpus.load(FilePathManager.resolve(args.corpus_path))
    print(corpus.word_from_index(0))

    dataset = CocoDataset(corpus, root=args.image_dir, annFile=args.test_path, transform=extractor.tf_image)

    # Build data loader
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    # Build the models
    # Build the models
    model = m_RNN(use_cuda=use_cuda,
                  image_regions=extractor.regions_count,
                  regions_features=extractor.regions_features_size,
                  features_size=extractor.features_size)

    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    # Load State Dictionary
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    start_word = torch.LongTensor([corpus.word_index('<start>')]).cuda()
    pred_captions = []

    for i, (image, _, _, img_id) in enumerate(dataloader):
        image = image.cuda()
        image_features, image_regions = extractor.forward(image)
        batch_size = image_features.shape[0]

        sampled_ids, _ = model.sample(image_features, image_regions, start_word, args.beam_size)
        sampled_ids = sampled_ids.cpu().data.numpy().T
        print('fin')
        for j in range(batch_size):
            sentence = ''
            words = []
            for k in sampled_ids[j]:
                word = corpus.word_from_index(k)

                if word == '<end>':
                    break

                if len(word) == 0:
                    continue
                words.append(word)

            for k in range(len(words)):
                sentence += (' ' if k != 0 and words[k] != ',' else '') + words[k]

            print(sentence)
            pred_captions.append({'image_id': int(img_id[j]), 'caption': sentence})
    language_eval(pred_captions, args.out_path, 'test', args.test_path)


def language_eval(input_data, savedir, split, ann_file):
    if not osp.exists(savedir):
        os.makedirs(args.model_path)

    if type(input_data) == str:  # Filename given.
        checkpoint = json.load(open(input_data, 'r'))
        preds = checkpoint
    elif type(input_data) == list:  # Direct predictions give.
        preds = input_data

    coco = COCO(ann_file)
    valids = coco.getImgIds()

    # Filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('Using %d/%d predictions' % (len(preds_filt), len(preds)))
    resFile = osp.join(savedir, 'result_%s.json' % (split))
    json.dump(preds_filt, open(resFile, 'w'))  # Serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # Create output dictionary.
    out = []
    for metric, score in cocoEval.eval.items():
        out.append({'metric': metric, 'score': score})

    evalFile = osp.join(savedir, 'evaluation.json')
    json.dump(out, open(evalFile, 'w'))  # Serialize to temporary json file. Sigh, COCO API...

    # Return aggregate and per image score.
    return out, cocoEval.evalImgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus_path', type=str, default='data/corpus.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--out_path', type=str, default='models',
                        help='path for vocabulary wrapper')
    parser.add_argument('--model_path', type=str, default='models/49/model-18.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--test_path', type=str,
                        default='D:/Datasets/mscoco/test/captions_train2017.json',
                        help='path for train annotation json file')
    parser.add_argument('--image_dir', type=str, default='D:/Datasets/mscoco/test/images',
                        help='directory for resized images')
    parser.add_argument('--image_regions', type=int, default=49,
                        help='number of image regions to be extracted (49 or 196) 64 for inception_v3')
    parser.add_argument('--beam_size', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()
    main(args)
