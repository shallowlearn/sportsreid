'''
Create model soup from multiple models

Author:

'''
import copy

import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchreid.utils.torchtools import resume_from_checkpoint
from tqdm import tqdm
def main(args):
    with open(args.f, 'r') as f:
        model_paths = f.readlines()
    model_paths = [i.strip() for i in model_paths]
    outdir = args.o
    if outdir is not None:
        os.makedirs(args.o, exist_ok=True)

    model_name = args.m
    img_size = None
    if model_name == 'vit_b_16':
        from torchreid.models.vit import vit_b_16 as modelclass
    elif model_name == 'deit_b_16_ls_224':
        from torchreid.models.deit import deit_b_16_ls as modelclass
        img_size = [224, 224]

    model_soup = None
    for idx, ckpt in tqdm(enumerate(model_paths)):
        # get number of classes
        x = torch.load(ckpt, 'cpu')
        num_classes = x['state_dict']['module.classifier.weight'].shape[0]
        x = None
        model = modelclass(num_classes=num_classes, pretrained=True, img_size=img_size)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        resume_from_checkpoint(ckpt, model, optimizer=None, scheduler=None)
        model.eval()
        if model_soup is None:
            # Deepcopy of state dict
            model_soup = copy.deepcopy(model.state_dict())
        else:
            new_state_dict = model.state_dict()
            # Iterate over keys
            for k in new_state_dict:
                new_val = model_soup[k].to(torch.double)
                new_val *= idx #Get back the prev sum from avg value
                # Add the new params
                new_val += new_state_dict[k].to(torch.double)
                # Compute running average
                new_val /= (idx + 1)
                model_soup[k] = new_val

    #Convert back to float32
    for k in model_soup:
        model_soup[k] = model_soup[k].to(torch.float32)

    # Save output soup
    outf = args.f.replace('.paths', '')
    outf = outf.replace('.txt', '.pth')
    if outdir is not None:
        outf = os.path.join(outdir, os.path.split(outf)[-1])

    torch.save({'state_dict': model_soup, 'epoch':None}, open(outf,'wb'))

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--f', required=False, type=str, default=None, help='')
    parser.add_argument('-o', '--o', required=False, type=str, default=None, help='outfolder')
    parser.add_argument('-m', '--m', required=False, type=str, default=None, help='model name')
    #parser.add_argument('-', '--', action='store_true', help='')
 
    args = parser.parse_args()
    main(args)