import argparse

import torch
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
from non_leaking import augment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def generate(args, loader, generator, encoder, device, mean_latent):
    loader = sample_data(loader)
    real_img = next(loader)[0]
    real_img = real_img.to(device)

    with torch.no_grad():
        generator.eval()

        style = encoder(real_img)
        sample, _ = generator(style, truncation=args.truncation, truncation_latent=mean_latent)
        
        utils.save_image(
            sample,
            f'infer/{str(i).zfill(6)}.png',
            nrow=args.sample//2,
            normalize=True,
            range=(-1, 1),
        )

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='/dataset/')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="checkpoint/021000.pt")
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

    dataset = AIDataset(
        name="base", 
        root_path=args.path,
        resolution=args.size
    )
    # dataset = MultiResolutionDataset(
    #     name="base", 
    #     root_path=osp.join(args.path, 'train'), 
    #     resolution=args.size, 
    #     domain=0, 
    #     output_type_lst=output_type_lst
    # )
    loader = data.DataLoader(
        dataset,
        batch_size=args.sample,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )
    generate(args, loader, generator, encoder, device, mean_latent)
