import argparse
import os.path as osp

import torch
from torch.nn import functional as F
from torchvision import utils
from model import Generator, pSpEncoder
from tqdm import tqdm
from dataset import *
from torch.utils import data
from distributed import (
    get_rank, 
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

transform_3ch = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
transform_1ch = transforms.Compose(
[
    transforms.ToTensor(),
])
def generate(args, generator, encoder, device, mean_latent):
    folders = os.listdir(args.path)
    for i in folders:
        real_img = Image.open(osp.join(args.path, i, 'image.png'))
        real_img = Image.fromarray(np.array(real_img)[:,:,:3])
        seg = Image.open(osp.join(args.path, i, 'seg.png'))

        real_img = transform_3ch(real_img)
        seg = transform_1ch(seg)
        
        real_img.unsqueeze_(0)
        seg.unsqueeze_(0)
        
        real_img = F.interpolate(real_img, size=1024, mode='bilinear')
        seg = F.interpolate(seg, size=1024, mode='bilinear')
        seg = np.array(seg)
        seg[seg>0] = 1
        seg = np.concatenate([seg, seg, seg], axis=1)
        seg = torch.tensor(seg)
        mask_img = real_img * seg

        mask_img = mask_img.to(device)

        with torch.no_grad():
            generator.eval()

            style = encoder(mask_img)
            sample, _ = generator(style, truncation=args.truncation, truncation_latent=mean_latent)
            
            concat = torch.stack([mask_img, sample])
            concat = concat.view(-1,3,1024,1024)
            os.makedirs('infer', exist_ok=True)
            utils.save_image(
                concat,
                f'infer/inp_{i}.png',
                nrow=args.sample,
                normalize=True,
                range=(-1, 1),
            )

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='test_inp/')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="checkpoint/ffhq-inpainting.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    encoder = pSpEncoder().to(device)
    checkpoint = torch.load(args.ckpt)

    generator.load_state_dict(checkpoint['g'])
    encoder.load_state_dict(checkpoint['e'])


    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, generator, encoder, device, mean_latent)
