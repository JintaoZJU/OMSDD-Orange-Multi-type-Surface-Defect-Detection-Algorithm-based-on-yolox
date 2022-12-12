from __future__ import print_function
import argparse
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from cnn import CNN

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from copy import deepcopy


def main():
    parser = argparse.ArgumentParser(
        description='',
        epilog=
        'Command example:\n>python test.py 0 --imgsize 128 --class_num 4 --data_dir data/jujube/test --pt_model ptModel.pt',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'model_type',
        type=int,
        metavar='N',
        choices=range(0, 3),
        help=
        'Choose a model\n0: model_a\n1: model_b\n2: model_c'
    )
    parser.add_argument('--imgsize',
                        type=int,
                        metavar='N',
                        required=True
                        )
    parser.add_argument('--class_num',
                        type=int,
                        metavar='N',
                        required=True,
                        choices=range(2, 33))
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--data_dir', help='test data directory')
    parser.add_argument('--pt_model', help='pytorch model file')
    args = parser.parse_args()

    test_data = args.data_dir
    pt_model = args.pt_model

    RESIZE = transforms.Resize((args.imgsize, args.imgsize))
    if args.model_type == 0:
        print('model a')
        in_ch = 1
        mode = 0
        c4_och = 16
        GRAY = transforms.Grayscale(in_ch)
        TENS = transforms.ToTensor()
        NORM = transforms.Normalize((0.0, ), (1.0, ))
        transf = transforms.Compose([RESIZE, GRAY, TENS, NORM])
    elif args.model_type == 1:
        print('model b')
        in_ch = 1
        mode = 1
        c4_och = 32
        GRAY = transforms.Grayscale(in_ch)
        TENS = transforms.ToTensor()
        NORM = transforms.Normalize((0.0, ), (1.0, ))
        transf = transforms.Compose([RESIZE, GRAY, TENS, NORM])
    elif args.model_type == 2:
        print('model c')
        in_ch = 1
        mode = 2
        c4_och = 32
        GRAY = transforms.Grayscale(in_ch)
        TENS = transforms.ToTensor()
        NORM = transforms.Normalize((0.0, ), (1.0, ))
        transf = transforms.Compose([RESIZE, GRAY, TENS, NORM])

    fc_inch = (int)(pow((args.imgsize / 16), 2) * c4_och)

    print('imgsize:', args.imgsize)
    print('class:', args.class_num)
    print('---------------------------------')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    model = CNN(in_ch, args.class_num, mode, fc_inch)
    origin_state_dict = torch.load(pt_model)
    model.load_state_dict(origin_state_dict)

    model.to(device)

    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=test_data, transform=transf),
            batch_size=1,
            shuffle=False,
            **kwargs)

    model.eval()
    nSoftmax = nn.Softmax(dim=1)
    correct = 0
    preds = []
    gts = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            smax = nSoftmax(output)

            preds.append(int(pred))
            gts.append(int(target))

            if (use_cuda):
                smax = smax.to("cpu")

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    print("-------------------------------------------")
    print("Test: {}/{} \t({:.2f}%)".format(correct, len(test_loader.dataset),
                                           test_acc))

    conf_mat = confusion_matrix(gts,
                                preds,
                                labels=[i for i in range(args.class_num)])
    print(conf_mat)
    conf_mat = pd.DataFrame(data=conf_mat,
                            index=[i for i in range(args.class_num)],
                            columns=[i for i in range(args.class_num)])
    sns.heatmap(conf_mat, square=True, annot=True, cmap='CMRmap', fmt='d')
    plt.yticks(rotation=0)
    plt.xlabel("Prediction", fontsize=13, rotation=0)
    plt.ylabel("Ground Truth", fontsize=13)
    plt.title(
        f'Inference Results: Type {args.model_type}, {args.imgsize}x, acc {test_acc:.2f} %'
    )
    plt.savefig(
        f'results/type_{args.model_type}_imgsize{args.imgsize}_acc{test_acc:.2f}.png')

    return


if __name__ == '__main__':
    main()
