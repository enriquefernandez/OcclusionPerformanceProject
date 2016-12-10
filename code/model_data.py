from __future__ import division
import attr
import numpy as np
import h5py
import os
import cv2

from synsets import load_syn_data, det_clsloc_names
from val_data import load_val_database, ValImage, ObjectAnnotation
from val_utils import get_db_image
from occlussion_generator import add_gaussian_blur_to_bb
from bounding_boxes import BoundingBox

results = "../results/"
bb_folder = "../bounding_boxes/"


FILTER_SIZE=(55, 55)
FILTER_SIGMA=55.5

MODEL_SIZES = {"resnet50": (224, 224),
               "inceptionv3": (299, 299),
               "vgg16": (224, 224),
               "vgg19": (224, 224),
               "alexnet": (227, 227)}

@attr.s
class ModelData(object):
    name = attr.ib()
    correct_mask_top1 = attr.ib(repr=False) # With no occlusions. Len is size of database
    correct_mask_top5 = attr.ib(repr=False)  # With no occlusions
    overall_top1_acc = attr.ib() # With no occlusions
    overall_top5_acc = attr.ib() # With no occlusions
    num = attr.ib() # Num images in the original dataset
    datasets = attr.ib(default=attr.Factory(dict), repr=False) # Dict indexed by name

@attr.s
class DataSet(object):
    name = attr.ib()
    results_file = attr.ib()
    bb_file = attr.ib()
    occlusion = attr.ib()
    all = attr.ib(repr=False)
    model = attr.ib(repr=False)
    categories = attr.ib(default=attr.Factory(dict), repr=False) # Dict indexed by category id, from 1 to 200


@attr.s
class CategoryData(object):
    idx = attr.ib() # Category index from 1 to 200 (except category all is -1)
    name = attr.ib()
    synset = attr.ib(repr=False)
    img_indexes = attr.ib(repr=False) # Indexes of all images in this category (regardless of results)
    correct_top1 = attr.ib(repr=False) # Mask of overall correct images in this category. Len is FULL size of db (this is a mask)
    correct_top5 = attr.ib(repr=False) # Mask of overall correct images in this category. (doesn't involve correct images in original model)
    category_mask = attr.ib(repr=False) # Mask of db images belonging to this category. Full size. For cat "all" this is all True
    missed_mask_top1 = attr.ib(repr=False) # Full size (len of db). Element True if image was correctly class in original, but not now
    missed_mask_top5 = attr.ib(repr=False)
    dataset = attr.ib(repr=False)

    def top1_acc(self, only_on_original_correct=False):
        num = self.num_top1(only_on_original_correct)
        if num == 0:
            return np.NAN

        if only_on_original_correct:
            original_correct = np.logical_and(self.category_mask, self.dataset.model.correct_mask_top1)
            correct_mask = np.logical_and(original_correct, self.correct_top1)
        else:
            correct_mask = self.correct_top1
        
        correct = np.sum(correct_mask)
        return correct/num

    def num_top1(self, only_on_original_correct=False):
        if only_on_original_correct:
            original_correct = np.logical_and(self.category_mask, self.dataset.model.correct_mask_top1)
            return np.sum(original_correct)
        else:
            return len(self.img_indexes)

    def top5_acc(self, only_on_original_correct=False):
        num = self.num_top5(only_on_original_correct)
        if num == 0:
            return np.NAN
        if only_on_original_correct:
            original_correct = np.logical_and(self.category_mask, self.dataset.model.correct_mask_top5)
            correct_mask = np.logical_and(original_correct, self.correct_top5)
        else:
            correct_mask = self.correct_top5

        correct = np.sum(correct_mask)
        return correct / num

    def num_top5(self, only_on_original_correct=False):
        if only_on_original_correct:
            original_correct = np.logical_and(self.category_mask, self.dataset.model.correct_mask_top5)
            return np.sum(original_correct)
        else:
            return len(self.img_indexes)


def get_prediction(data_set, img_idx):
    """Returns DET and CLSLOC top 5 prediction.
    List of 5 tuples (det_cat_id, clsloc_cat_id)"""
    f_h5 = h5py.File(data_set.results_file)
    det = f_h5["top5_det"][img_idx, :]
    clsloc = f_h5["top5_clsloc"][img_idx, :]
    pct = f_h5["top5_preds"][img_idx, :]
    f_h5.close()
    return list(zip(det,clsloc, pct))


def get_readable_prediction(data_set, syn_data, img_idx):
    return [(det_clsloc_names(syn_data, d,c),p) for d,c,p in get_prediction(data_set, img_idx)]

def get_dataset_img(val_db, data_set, img_idx, show_bb=True, load_size='original'):
    # Load image
    img = get_db_image(val_db.val_data_single[img_idx])

    bb_db_f = data_set.bb_file
    if bb_db_f:
        # Load noise bb and add noise
        f_h5 = h5py.File(bb_db_f)
        bb_db = f_h5["bounding_boxes"]
        noise_bb = BoundingBox(*bb_db[img_idx])
        img = add_gaussian_blur_to_bb(img, noise_bb,
                                      filter_size=FILTER_SIZE, filter_sigma=FILTER_SIGMA)
    if show_bb:
        ob = val_db.val_data_single[img_idx].objects[0]
        minx, miny, maxx, maxy = ob.xmin, ob.ymin, ob.xmax, ob.ymax
        cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
        if bb_db_f:
            minx, miny, maxx, maxy = noise_bb.to_tuple()
            cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)

    if isinstance(load_size, str):
        if load_size.lower() == "original":
            pass
        elif load_size.lower() == "model":
            img = cv2.resize(img, MODEL_SIZES[data_set.model.name])
        else:
            raise ValueError("wrong size spec")
    else:
        img = cv2.resize(img, load_size)

    return img


def load_model_results(syn_data, db):
    model_names = ["resnet50", "inceptionv3", "vgg16", "vgg19", "alexnet"]
    data_sets = ["original"] + \
                ["bb_center_010", "bb_center_020", "bb_center_050", "bb_center_075", "bb_center_100"] + \
                ["bb_corner_010", "bb_corner_020", "bb_corner_050", "bb_corner_075"] + \
                ["bb_random_010", "bb_random_020", "bb_random_050", "bb_random_075"]

    models = dict()

    for mn in model_names:
        print("Processing model {}".format(mn))

        for i, ds in enumerate(data_sets):
            # print("\tProcessing model: {} dataset: {} ({}/{})".format(mn, ds, i+1, len(data_sets)))
            results_file = os.path.join(results, "preds_{}_{}.h5".format(mn, ds))
            # print("\tOpening file {}".format(results_file))
            f_h5 = h5py.File(results_file)

            # Load correct data for this dataset
            correct_mask_top1 = np.array(f_h5["correct_top1"][:,0])
            correct_mask_top5 = np.array(f_h5["correct_top5"][:,0])
            assert len(correct_mask_top1) == len(correct_mask_top5)
            num = len(correct_mask_top1)
            # print("num: ",num)

            if i==0:
                # Process original model, to compute indices of correctly classified images in original images
                model = ModelData(name=mn,
                                  correct_mask_top1=correct_mask_top1,
                                  correct_mask_top5=correct_mask_top5,
                                  overall_top1_acc=np.sum(correct_mask_top1) / num,
                                  overall_top5_acc=np.sum(correct_mask_top5) / num,
                                  num=num)

                pass
            else:
                # Deal with other occluded datasets
                pass
            # Create dataset
            if ds=="original":
                bb_file = None
            else:
                bb_file = os.path.join(bb_folder, "{}.h5".format(ds))
            data = DataSet(name=ds, results_file=results_file,
                           bb_file=bb_file,
                           occlusion=-1, # TODO: change this
                           all=None, model=model)
            # Process overall + 200 categories
            # Overall category is everything
            # But different for each dataset!
            # num_correct_top1 = np.sum(correct_mask_top1)
            # num_correct_top5 = np.sum(correct_mask_top5)
            missed_mask_top1 = np.logical_and(model.correct_mask_top1,
                                              np.logical_not(correct_mask_top1))
            missed_mask_top5 = np.logical_and(model.correct_mask_top5,
                                              np.logical_not(correct_mask_top5))

            all_cat = CategoryData(idx=-1, name="all", synset=None,
                                   img_indexes=np.array(range(db.num_single)),
                                   correct_top1=correct_mask_top1,
                                   correct_top5=correct_mask_top5,
                                   category_mask=np.ones(correct_mask_top1.shape, dtype='bool'),
                                   # Previously correct but now incorrect ones...
                                   missed_mask_top1=missed_mask_top1,
                                   missed_mask_top5=missed_mask_top5,
                                   dataset=data
                                   )
            data.all = all_cat
            # Other categories
            # print("\tProcessing {}/{} all 200 categories".format(mn, ds))
            for j in range(200):
                # print("Cat: ",j)
                cat_id = j+1 # From 1 to 200
                synset = syn_data.det_synsets[syn_data.det_synsets_id[cat_id]]
                name = synset["name"]
                cat_info = db.det_categories[cat_id]
                cat_mask = cat_info.index_mask
                # print(cat_mask.dtype)
                cat_correct_top1 = np.logical_and(cat_mask, correct_mask_top1)
                # print("Len catmask: {}, len correct mask: {}".format(cat_mask.shape, correct_mask_top5.shape))
                cat_correct_top5 = np.logical_and(cat_mask, correct_mask_top5)

                cat_missed_mask_top1 = np.logical_and(
                    np.logical_and(cat_mask, model.correct_mask_top1),
                    np.logical_not(cat_correct_top1))
                cat_missed_mask_top5 = np.logical_and(
                    np.logical_and(cat_mask, model.correct_mask_top5),
                    np.logical_not(cat_correct_top5))
                cat = CategoryData(idx=cat_id, name=name, synset=synset,
                                   img_indexes=cat_info.indexes,
                                   correct_top1=cat_correct_top1,
                                   correct_top5=cat_correct_top5,
                                   category_mask=cat_mask,
                                   # Previously correct in this cat but now incorrect ones...
                                   missed_mask_top1=cat_missed_mask_top1,
                                   missed_mask_top5=cat_missed_mask_top5,
                                   dataset=data
                                   )
                data.categories[cat_id] = cat

            model.datasets[ds] = data
            f_h5.close()
            # print("\tDONE processing {}/{}!!".format(mn, ds))

        models[mn] = model
        print("DONE! Processing model {}".format(mn))

    print("All models processed!!")
    return models











if __name__ == "__main__":
    syn_data = load_syn_data()
    db = load_val_database(syn_data)
    models = load_model_results(syn_data, db)
