import os
import glob
import numpy as np
import random
from PIL import Image, ImageEnhance

def get_coords(hr_size, lr_size, scale):

    scale_hw = (float(hr_size[0])/lr_size[0], float(hr_size[1])/lr_size[1])

    coords = np.mgrid[0:hr_size[0], 0:hr_size[1]]
    coords = coords.astype("float32")
    coords = np.transpose(coords, [1, 2, 0])

    coords[:,:,0] = (coords[:,:,0]/scale_hw[0])%1
    coords[:,:,1] = (coords[:,:,1]/scale_hw[1])%1

    coords = np.concatenate([
        coords,
        np.ones((hr_size[0], hr_size[1],1),"float32")/scale
    ],axis=-1)
    return coords

def data_loader(folder, max_scale, batch_size, patch_size, augmentation, preload_all_image):
    image_paths = glob.glob(os.path.join(folder, "*.png"))

    if preload_all_image:
        hr_images = []
        for path in image_paths:
            hr_images.append(Image.open(path).convert("RGB"))

    while True:
        rand_idx = np.random.permutation(len(image_paths))[:batch_size]

        batch_hr_images = []
        batch_lr_images = []
        batch_coords = []

        rand_scale = random.uniform(1.0, max_scale)

        for i in rand_idx:
            if preload_all_image:
                hr_image = hr_images[i]
            else:
                hr_image = Image.open(image_paths[i]).convert("RGB")

            crop_left = random.randint(0, hr_image.size[0] - patch_size - 1)
            crop_top = random.randint(0, hr_image.size[1] - patch_size - 1)
            hr_image = hr_image.crop((crop_left, crop_top, crop_left + patch_size, crop_top + patch_size))

            if augmentation:
                hr_image = hr_image.transpose(Image.FLIP_TOP_BOTTOM) if np.random.randint(0, 2) is 0 else hr_image
                rand_rot = np.random.randint(1, 4)
                hr_image = hr_image.transpose(rand_rot) if rand_rot != 1 else hr_image
                hr_image = ImageEnhance.Color(hr_image).enhance(random.uniform(0.0, 2.0))
                hr_image = ImageEnhance.Contrast(hr_image).enhance(random.uniform(0.75, 1.25))
                hr_image = ImageEnhance.Brightness(hr_image).enhance(random.uniform(0.75, 1.25))

            hr_w, hr_h = hr_image.size
            lr_w, lr_h = int(hr_image.size[0] / rand_scale), int(hr_image.size[1] / rand_scale)
            lr_image = hr_image.resize((lr_w, lr_h), Image.BICUBIC)

            hr_image = np.asarray(hr_image, 'float32') / 127.5 - 1
            lr_image = np.asarray(lr_image, 'float32') / 127.5 - 1

            batch_lr_images.append(lr_image)
            batch_hr_images.append(hr_image)
            batch_coords.append(get_coords((hr_h, hr_w), (lr_h, lr_w), rand_scale))

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images)
        batch_coords = np.asarray(batch_coords)
        yield [batch_lr_images, batch_coords], [batch_hr_images]