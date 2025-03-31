import argparse
import math
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from config import Config
from hlibf_vit import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224

parser = argparse.ArgumentParser(description='Hardware Fitting Quantization Framework')

parser.add_argument('model', choices=['deit_tiny', 'deit_small', 'deit_base'], help='model to be quantized: deit_tiny/small/base')
parser.add_argument('--data', default='/vol/datastore/imagenet', help='path to dataset')
parser.add_argument('--quant', default=False, action='store_true')
parser.add_argument('--calib-batchsize', default=20, type=int, help='batchsize of calibration set (default: 100)')
parser.add_argument('--val-batchsize', default=100, type=int, help='batchsize of validation set (default: 100)')
parser.add_argument('--num-workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--seed', default=0, type=int, help='seed (default: 0)')

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def str2model(name):
    md = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
    }
    print('Model: %s' % md[name].__name__)
    return md[name]

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.format(top1=top1, top5=top5, time=val_end_time - val_start_time))

def accuracy(output, target, topk=(1, )):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def build_transform(input_size=224, interpolation='bicubic', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(size, interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)
    model = str2model(args.model)(pretrained=True, cfg=Config())
    model = model.to(device)

    model_type = args.model.split('_')[0]
    if model_type == 'deit':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    else:
        raise NotImplementedError

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    if args.quant:
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.calib_batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
        print('Calibrating...')
        model.model_open_calibrate()
        with torch.no_grad():
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                model.model_open_last_calibrate()
                model(data)
                break
        model.model_close_calibrate()
        model.model_quant()

    print('Validating...')
    validate(args, val_loader, model, criterion, device)

if __name__ == '__main__':
    main()
