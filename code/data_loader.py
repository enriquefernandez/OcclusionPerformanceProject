from __future__ import division
# from models.resnet50 import ResNet50
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import h5py
import json

from val_utils import get_db_image
from occlussion_generator import add_gaussian_blur_to_bb
from bounding_boxes import BoundingBox

class ValDataLoader(object):
    def __init__(self, db, target_size, bb_db_file=None, filter_size=(55, 55), filter_sigma=55.5,
                 preproc_fun=None, img_order='tf'):
        if preproc_fun is None:
            self.preproc_fun = preprocess_input
        else:
            self.preproc_fun = preproc_fun # For inception model
        self.img_order = img_order
        self.db = db
        self.single_db = db.val_data_single
        self.target_size = target_size
        self.bb_db = None
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma

        self.num = len(self.single_db)
        self._idx = 0

        if bb_db_file:
            # Load bb database
            bb_h5 = h5py.File(bb_db_file)
            self.bb_db = bb_h5["bounding_boxes"]
            assert self.num == len(self.bb_db)


    def reset(self):
        self._idx = 0

    def num_batches(self, batch_size):
        num_batches = int(np.ceil(self.num / batch_size))
        return num_batches

    def get_bb(self, idx):
        if self.bb_db:
            return BoundingBox(*self.bb_db[idx])
        else:
            return None

    def prepare_img(self, idx, add_noise=True, show_bb=False):
        # Load image
        img = get_db_image(self.single_db[idx])

        if self.bb_db and add_noise:
            # Apply noise according to specs
            noise_bb = self.get_bb(idx)
            img = add_gaussian_blur_to_bb(img, noise_bb,
                                          filter_size=self.filter_size, filter_sigma=self.filter_sigma)
            if show_bb:
                # Show rectangles
                ob = self.single_db[idx].objects[0]
                minx, miny, maxx, maxy = ob.xmin, ob.ymin, ob.xmax, ob.ymax
                cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
                minx, miny, maxx, maxy = noise_bb.to_tuple()
                cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)

        # Resize image and prepare
        img = cv2.resize(img, self.target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preproc_fun(x)
        return x, img

    def prepare_batch(self, batch_size):
        if self._idx + batch_size >= self.num:
            batch_size = self.num - self._idx
        width, height = self.target_size
        if self.img_order == 'th':
            batch = np.zeros((batch_size, 3, width, height))
        else:
            batch = np.zeros((batch_size, width, height, 3))
        # print batch_idx
        # for i, idx in enumerate(self.db[self._idx:self._idx + batch_size]):
        for i in range(batch_size):
            idx = self._idx + i
            x, img = self.prepare_img(idx, add_noise=True, show_bb=False)
            batch[i, ...] = x

        self._idx += batch_size
        # print("Batch size: ", batch.shape)
        return batch
