import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.pjrt_backend

from torch.utils.data import DataLoader
from torchvision import transforms

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
from torch_xla.experimental import pjrt
from einops import rearrange

import numpy as np
from time import time
from models import DiT_models
import os
from PIL import Image
import logging
import argparse
import glob
from copy import deepcopy

from coco import CocoDataset, collate_fn
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from transformers import DistilBertModel

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

@torch.no_grad()
def text_encoding(caption, encoder, end_token, dropout=None):
    device = caption.device
    mask = torch.cumsum((caption == end_token), 1).to(device)
    mask[caption == end_token] = 0
    mask = (~mask.bool()).long()

    emb = encoder(caption, attention_mask=mask)['last_hidden_state']
    
    emb = rearrange(emb, 'b c h -> b (c h)')

    return emb

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if xm.is_master_ordinal():  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args, encoder):
    device = xm.xla_device()
    torch.manual_seed(42)
    print(f"Starting on {device}.")

    # Setup an experiment folder:
    # if xm.is_master_ordinal():
    #     os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    #     experiment_index = len(glob(f"{args.results_dir}/*"))
    #     model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    #     experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    #     checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     logger = create_logger(experiment_dir)
    #     logger.info(f"Experiment directory created at {experiment_dir}")
    # else:
    #     logger = create_logger(None)

    model = DiT_models[args.model](
        input_size=32,
        num_classes=1000,
        emb_dropout_prob=0.0,
    ).to(device)
    
    if pjrt.using_pjrt():
        pjrt.broadcast_master_param(model)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule, MSE loss
    test_diffusion = create_diffusion(str(250)) # for sampling
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    
    xm.rendezvous('VAE loaded') # probably need to separate vae and encoder loaded

    # 

    xm.master_print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    assert os.path.isdir(args.data_path), f'Could not find COCO2017 at {args.data_path}'

    def img_augment(data):
        return center_crop_arr(data, args.image_size)

    train_transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        img_augment,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train_path = os.path.join(args.data_path, 'train2017')
    train_ann_path = os.path.join(args.data_path, 'annotations/captions_train2017.json')

    train_dataset = CocoDataset(train_path, train_ann_path, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    mp_device_loader = pl.MpDeviceLoader(
        train_loader,
        device,
    )

    xm.rendezvous('finish loading')

    print(f"On {xm.get_local_ordinal()}, Dataset contains {len(train_dataset):,} images ({args.data_path})")
    
    model.train()

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    xm.master_print(f"Training for 1 epochs...")

    for epoch in range(1):
        xm.master_print(f"Beginning epoch {epoch}...")
        for x, y in mp_device_loader:
            x = x.to(device)
            y = y
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            y = text_encoding(y, encoder, 102).to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y) # class conditional
            # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            xm.optimizer_step(opt)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0 and train_steps > 0:
                # Measure training speed:
                # Synchornize
                xm.rendezvous('log loss')
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device).item()
                avg_loss = xm.mesh_reduce('training_loss', avg_loss, np.mean)
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                xm.master_print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()


def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')

    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

    main(args, encoder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="t2i-results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--fine-tuning", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    args = parser.parse_args()

    os.environ['PJRT_DEVICE'] = 'TPU'

    xmp.spawn(_mp_fn, args=(args, ))