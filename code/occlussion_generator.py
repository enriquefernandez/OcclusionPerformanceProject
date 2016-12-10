import cv2
import numpy as np

def add_gaussian_blur_to_bb(img, noise_bb, filter_size=(55,55), filter_sigma=55.5):

    img_blur = np.array(img)
    minx, miny, maxx, maxy = noise_bb.to_tuple()
    img_rect = img[miny:maxy, minx:maxx]

    img2 = cv2.GaussianBlur(img_rect, filter_size, filter_sigma)

    img_blur[miny:maxy, minx:maxx] = img2

    return img_blur
