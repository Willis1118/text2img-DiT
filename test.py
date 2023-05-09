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

import numpy as np
from time import time
from models import DiT_models
import os
from PIL import Image
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

def main():
    device = xm.xla_device()
    torch.manual_seed(42)
    print(f"Starting on {device}.")
    
    model = DiT_models['DiT-S/2'](
        input_size=32,
        num_classes=1000,
        emb_dropout_prob=0.0,
    ).to(device)
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule, MSE loss
    test_diffusion = create_diffusion(str(250)) # for sampling
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    xm.master_print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    def img_augment(data):
        return center_crop_arr(data, 256)

    train_transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        img_augment,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    data_path = '/mnt/disks/will-coco/coco'

    train_path = os.path.join(data_path, 'train2017')
    train_ann_path = os.path.join(data_path, 'annotations/captions_train2017.json')

    train_dataset = CocoDataset(train_path, train_ann_path, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    mp_device_loader = pl.MpDeviceLoader(
        train_loader,
        device,
    )

    # xm.rendezvous('finish loading')

    print(f"Dataset contains {len(train_dataset):,} images ({data_path})")
    
    # model.train()

    # # Variables for monitoring/logging purposes:
    # train_steps = 0
    # log_steps = 0
    # running_loss = 0
    # start_time = time()

    # xm.master_print(f"Training for 1 epochs...")

    # for epoch in range(1):
    #     xm.master_print(f"Beginning epoch {epoch}...")
    #     for x, y in mp_device_loader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         with torch.no_grad():
    #             # Map input images to latent space + normalize latents:
    #             x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    #         y = text_encoding(y, encoder, 102)
    #         t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    #         model_kwargs = dict(y=y) # class conditional
    #         # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

    #         loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    #         loss = loss_dict["loss"].mean()
    #         opt.zero_grad()
    #         loss.backward()
    #         xm.optimizer_step(opt)

    #         # Log loss values:
    #         running_loss += loss.item()
    #         log_steps += 1
    #         train_steps += 1

    #         if train_steps % 10 == 0 and train_steps > 0:
    #             # Measure training speed:
    #             # Synchornize
    #             xm.wait_device_ops()
    #             end_time = time()
    #             steps_per_sec = log_steps / (end_time - start_time)
    #             # Reduce loss history over all processes:
    #             avg_loss = torch.tensor(running_loss / log_steps, device=device).item()
    #             avg_loss = xm.mesh_reduce('training_loss', avg_loss, np.mean)
    #             # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
    #             xm.master_print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
    #             # Reset monitoring variables:
    #             running_loss = 0
    #             log_steps = 0
    #             start_time = time()


def _mp_fn(index):
    main()

def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
      device,
      step,
      loss,
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)

if __name__ == '__main__':
    os.environ['PJRT_DEVICE'] = 'TPU'
    xmp.spawn(_mp_fn)