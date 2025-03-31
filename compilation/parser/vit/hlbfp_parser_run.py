import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from hlbfp_vit import *
from utils import ( get_fc_layer_idx, get_softmax_layer_idx,)
from format_config import FormatConfig
from para_config import *
from parser.model_parser import ModelParser

parser = argparse.ArgumentParser(description="BFloat16 Quantization Transformer Framework")
parser.add_argument(
    "model",
    choices=["deit_tiny", "deit_small", "deit_base"],
    help="model to be quantized: deit_tiny/small/base",
)
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--data", default="/vol/datastore/imagenet", help="path to dataset")
parser.add_argument(
    "--val-batchsize",
    default=16,
    type=int,
    help="batchsize of validation set (default: 100)",
)
parser.add_argument(
    "--num-workers",
    default=32,
    type=int,
    help="number of data loading workers (default: 32)",
)
parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--seed", default=0, type=int, help="seed (default: 0)")
parser.add_argument("--dim", type=int, default=0)
parser.add_argument("--scale", type=float, default=0)

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

def str2model(name):
    md = {
        "deit_tiny": deit_tiny_patch16_224,
        "deit_small": deit_small_patch16_224,
        "deit_base": deit_base_patch16_224,
    }
    return md[name]

def model_parse(args, val_loader, device):
    print("********** Parser Starts **********")
    new_config = FormatConfig() # Create configuration
    model_format_config, model_format_list = new_config.init_format(7, 4, 3, 7) # Initialize the format configuration, -1 refers to FP32

    fc_idx_list = get_fc_layer_idx("deit", 12)
    softmax_idx_list = get_softmax_layer_idx("deit", 12)
    for widx in fc_idx_list:
        model_format_list[widx][1][0] = BLOCK_SIZE

    # Change some activation formats as block fp, except softmax
    for idx in range(len(model_format_list)):
        if len(model_format_list[idx]) == 4:  # act only
            if idx not in softmax_idx_list:
                model_format_list[idx][0] = BLOCK_SIZE
                model_format_list[idx][2] = 7
        elif len(model_format_list[idx]) == 3:  # act, weight and bias
            if idx not in softmax_idx_list:
                model_format_list[idx][0][0] = BLOCK_SIZE
                model_format_list[idx][0][2] = 7
                model_format_list[idx][2][0] = 8
                model_format_list[idx][2][2] = 7

    # embed layer
    model_format_list[0][0][0] = BLOCK_SIZE # Embed Conv2d Act
    model_format_list[0][0][2] = 7
    model_format_list[0][1][0] = BLOCK_SIZE # Embed Conv2d Weight
    model_format_list[0][1][2] = 7
    model_format_list[0][2][0] = 8 # Embed Conv2d Bias
    model_format_list[0][2][2] = 7

    # classifier layer
    model_format_list[-1][0][0] = BLOCK_SIZE
    model_format_list[-1][0][2] = 7
    model_format_list[-1][1][0] = BLOCK_SIZE
    model_format_list[-1][1][2] = 7
    model_format_list[-1][2][0] = 8
    model_format_list[-1][2][2] = 7

    # Change the softmax layer format, from block 2 to block 11
    for sftm_idx in range(0, 12):
        model_format_list[1 + 2 + sftm_idx * 7][0] = BLOCK_SIZE
        model_format_list[1 + 2 + sftm_idx * 7][2] = 7
    model_format_config = new_config.config_list_to_dict(model_format_list)

    # Finally, generate the BFP format for each layer
    model_format = new_config.config_dict_to_model_format(model_format_config)

    # Load the model and proceed to inference
    model = str2model(args.model)(pretrained=True, model_format=model_format)
    mp = ModelParser(args.model, args.val_batchsize)

    C_dict = {'C0':0.0, 'C1':0.0, 'C2':0.0, 'C3':0.0, 'C4':0.0, 'C5':0.0, 'C6':0.0, 'C7':0.0}

    model = model.to(device)
    model.eval()

    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        print("Data Features: ", data.shape)
        target = target.to(device)

        with torch.no_grad():
            for i in range(10000):
                model(data, mp, C_dict)
        break

    ms = mp.return_ms()    
    ms.update_param('constants', C_dict)
    ms.save(f"./model_spec/{args.model}_bs{args.val_batchsize}.json")
 
    print("********** Parser Finishes **********")

def build_transform(
    input_size=224,
    interpolation="bicubic",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    crop_pct=0.875,
):
    def _pil_interp(method):
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 64
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size, interpolation=ip
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def main():
    args = parser.parse_args()
    device = torch.device(args.device)
    model_type = args.model.split("_")[0]
    if model_type == "deit":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    else:
        raise NotImplementedError

    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    valdir = os.path.join(args.data, "val")

    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model_parse(args, val_loader, device)

if __name__ == "__main__":
    main()
