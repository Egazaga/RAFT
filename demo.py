import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from .core.raft import RAFT
from .core.utils import flow_viz
from .core.utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def convert(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return img_flo[:, :, [2, 1, 0]] / 255.0


def infer_raft(in_path, out_path):
    args = {'model': './models/raft-things.pth',
            'path': in_path,
            'small': False,
            'mixed_precision': False,
            'alternate_corr': False}

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(in_path, '*.png')) + \
                 glob.glob(os.path.join(in_path, '*.jpg'))

        images = sorted(images)
        for i, imfile1, imfile2 in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            cv2.imwrite(out_path + str(i).zfill(6) + ".png", convert(image1, flow_up))


if __name__ == '__main__':
    infer_raft(in_path="in", out_path="out")
