from __future__ import division
import time
import h5py
import os
from bounding_boxes import BoundingBox, center_bounding, corner_bounding_box, random_bounding_box
from val_data import load_val_database, ValDatabase, ValImage, ObjectAnnotation
from synsets import load_syn_data
import numpy as np

bb_folder = "../bounding_boxes/"

def generate_center_bboxes(db, occlusion_pct):
    out_file = "bb_center_{:03d}.h5".format(int(occlusion_pct * 100))
    h5_name = os.path.join(bb_folder, out_file)
    f_h5 = h5py.File(h5_name, "w")
    print("Opened {} h5 file".format(h5_name))

    num_single = len(db)

    bb_dataset = f_h5.create_dataset("bounding_boxes", (num_single, 4), dtype='i')
    for i, val in enumerate(db):
        ob = val.objects[0]
        noise_bb = center_bounding((ob.xmin, ob.ymin, ob.xmax, ob.ymax), occlusion_pct)
        bb_dataset[i, ...] = noise_bb.to_tuple()
    f_h5.close()

def generate_corner_bboxes(db, occlusion_pct):
    out_file = "bb_corner_{:03d}.h5".format(int(occlusion_pct * 100))
    h5_name = os.path.join(bb_folder, out_file)
    f_h5 = h5py.File(h5_name, "w")
    print("Opened {} h5 file".format(h5_name))

    num_single = len(db)

    bb_dataset = f_h5.create_dataset("bounding_boxes", (num_single, 4), dtype='i')
    for i, val in enumerate(db):
        ob = val.objects[0]
        noise_bb = corner_bounding_box(np.random.randint(0,4),
                                       (ob.xmin, ob.ymin, ob.xmax, ob.ymax), occlusion_pct)
        bb_dataset[i, ...] = noise_bb.to_tuple()
    f_h5.close()

def generate_random_bboxes(db, occlusion_pct):
    out_file = "bb_random_{:03d}.h5".format(int(occlusion_pct * 100))
    h5_name = os.path.join(bb_folder, out_file)
    f_h5 = h5py.File(h5_name, "w")
    print("Opened {} h5 file".format(h5_name))

    num_single = len(db)

    bb_dataset = f_h5.create_dataset("bounding_boxes", (num_single, 4), dtype='i')
    for i, val in enumerate(db):
        ob = val.objects[0]
        noise_bb = random_bounding_box((ob.xmin, ob.ymin, ob.xmax, ob.ymax), occlusion_pct)
        bb_dataset[i, ...] = noise_bb.to_tuple()
    f_h5.close()

if __name__ == "__main__":
    syn_data = load_syn_data()
    db = load_val_database(syn_data).val_data_single
    # for opct in [0.1, 0.2, 0.5, 0.75, 1.00]:
    #     print("Generating bb database for center occlusion with pct={}".format(opct))
    #     generate_center_bboxes(db, opct)

    for opct in [0.1, 0.2, 0.5, 0.75]:
        print("Generating bb database for corner occlusion with pct={}".format(opct))
        np.random.seed(4231)
        generate_corner_bboxes(db, opct)

    for opct in [0.1, 0.2, 0.5, 0.75]:
        print("Generating bb database for random occlusion with pct={}".format(opct))
        np.random.seed(2312)
        generate_random_bboxes(db, opct)


    print("DONE!")