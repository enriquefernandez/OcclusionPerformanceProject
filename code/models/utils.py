import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import h5py
import json

from bounding_boxes import BoundingBox, center_bounding
from val_utils import nice_preds, classify_img


def example_img_center_blur(model, syn_data, val_data, idx, pct=.5):

    val = val_data[idx]
    img_path = val.img_path
    img = cv2.imread(img_path)

    img_blur = np.array(img)

    ob = val.objects[0]
    print("Object {}: {}, {}".format(i, ob.wnid, ob.synset["name"]))
    noise_bb = center_bounding((ob.xmin, ob.ymin, ob.xmax, ob.ymax), pct)
    minx, miny, maxx, maxy = noise_bb
    img_rect = img[miny:maxy, minx:maxx]

    img2 = cv2.GaussianBlur(img_rect, (55, 55), 55.5)

    img_blur[miny:maxy, minx:maxx] = img2

    #     # Classify original
    res_ori = nice_preds(classify_img(model, syn_data, img))
    res_ori = res_ori[0]

    #     # Classify blurred
    res_blur = nice_preds(classify_img(model, syn_data, img_blur))
    res_blur = res_blur[0]

    # Show rectangles
    minx, miny, maxx, maxy = ob.xmin, ob.ymin, ob.xmax, ob.ymax
    cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
    minx, miny, maxx, maxy = noise_bb.to_tuple()
    cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)

    print("\nOriginal predictions:\n DET: {}\n CLSLOC: {}\n".format(" ".join(res_ori[0]),
                                                                    " ".join(res_ori[1]), ))
    print("Blurred predictions:\n DET: {}\n CLSLOC: {}".format(" ".join(res_blur[0]),
                                                               " ".join(res_blur[1]), ))

    # plt.imshow(img)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    #     plt.title(" ".join(res_ori[0]))

    plt.subplot(1, 2, 2)
    plt.imshow(img_blur)
    #     plt.title(" ".join(res_blur[0]))

    plt.suptitle("Ground truth: {} ({})".format(ob.synset["name"], ob.wnid))