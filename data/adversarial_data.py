"""
Add random noise to images.
Rotate images by a random angle.

Parameters:
- images: Array of input images.
- scale: Scale of the random noise.

Returns:
- noisy_images: Array of images with added random noise.
- rotated_images: Array of rotated images.
"""


import numpy as np
import random
import cv2


def add_random_noise(images, scale):
    noise = np.random.normal(loc=0.0, scale=scale, size=images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images


def rotate_images(images, start_angle, end_angle):
    rotated_images = []
    for img in images:
        # Generate random angle between -30 and 90 degrees
        angle = random.randint(start_angle, end_angle)
        # Rotate image by random angle clockwise
        M = cv2.getRotationMatrix2D((14, 14), angle, 1) 
        rotated_img = cv2.warpAffine(img, M, (32, 32))
        rotated_images.append(rotated_img)

    rotated_images = np.array(rotated_images)
    return rotated_images
