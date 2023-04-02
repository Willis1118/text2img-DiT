from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

from torchvision import transforms

from coco import CocoDataset, collate_fn


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="samples/coco_val")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    args = parser.parse_args()

    eval_dir = f"{args.sample_dir}/coco_{args.image_size}"

    os.makedirs(eval_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    ])

    dataset = CocoDataset('/scratch/nm3607/datasets/coco/val2017', '/scratch/nm3607/datasets/coco/annotations/captions_val2017.json', transform=transform)
    iteration = 0
    for index, (img, cap) in tqdm(enumerate(dataset)):
        iteration += 1
        img.save(f"{eval_dir}/{index:06d}.png")
    
    create_npz_from_sample_folder(eval_dir, iteration)