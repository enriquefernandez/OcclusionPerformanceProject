import attr
import pickle
import xml.etree.ElementTree as ET
from utils import is_python2
from synsets import load_syn_data
import numpy as np
from scipy.special import comb

val_img_folder = "../ILSVRC2013_DET_val/"
val_annot_folder = "../ILSVRC2013_DET_bbox_val/"
val_idx_path = "../ILSVRC2014_devkit/data/det_lists/val.txt"
val_labels_path = "val_labels.txt"

val_pickle_file = "val_data_py2.pickle" if is_python2() else "val_data_py3.pickle"

@attr.s
class ValDatabase(object):
    val_data = attr.ib()
    val_data_single = attr.ib()
    single_idx = attr.ib()
    num_single = attr.ib()
    labels = attr.ib() # Labels of the single object db (7706 obj)
    det_categories = attr.ib() # Dictionary. Category id: from 1 to 200
    random_chance_top1 = attr.ib()
    random_chance_top5 = attr.ib()

@attr.s
class ObjectAnnotation(object):
    wnid = attr.ib()
    synset = attr.ib()
    xmin = attr.ib()
    xmax = attr.ib()
    ymin = attr.ib()
    ymax = attr.ib()

@attr.s
class ValImage(object):
    idx = attr.ib()
    img_path = attr.ib()
    objects = attr.ib()
    width = attr.ib()
    height = attr.ib()

@attr.s
class DetCategory(object):
    data_single = attr.ib(repr=False) # Pointer to parent db
    idx = attr.ib()
    wnid = attr.ib()
    name = attr.ib()
    desc = attr.ib(repr=False)
    num_clsloc_cats = attr.ib()  # Number of clsloc categories that map to this DET one
    random_chance_top1 = attr.ib()  # Random chance of guessing this det category
    random_chance_top5 = attr.ib()
    indexes = attr.ib(default=attr.Factory(np.array), repr=False) # Indexes of items of this det cat in single object db
    index_mask = attr.ib(default=attr.Factory(np.array), repr=False) # Mask of category indexes. Same size as all database size
    num = attr.ib(default=0) # Num of images in single object db in this category


    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)):
            # print("Detected slice: ", keys)
            idx_in_all = self.indexes[keys]
            # print("Computed all indexes: ", idx_in_all)
            return [self.data_single[idx] for idx in idx_in_all]
        else:
            return ValueError("Wrong indexing, not supported in my DetCategory class")



    def all(self):
        return self[:]



def det_prob(nc, top=5):
    N = 1000
    correct_combs = [comb(nc, i + 1, exact=False) * comb(N - nc, top - (i + 1), exact=False)
                     for i in range(top)]

    return np.sum(correct_combs) / comb(N, top, exact=False)

def prob_without_clsloc2det(top=5):
    return comb(200-1,top-1)/comb(200,top)

def load_val_database(syn_data):
    val_data = load_val_data()
    val_data_single, single_idx, num_single = generate_one_object_database(val_data)

    # Load ground truth
    print("Loading ground truth labels...")
    with open(val_labels_path) as f:
        labels_lines = f.read().split('\n')
        labels = np.zeros(num_single, dtype='i')
        for i,l in enumerate(labels_lines):
            line = l.strip()
            if line:
                split_line = line.split(" ")
                # print split_line
                labels[i] = int(split_line[1])
    print("Loaded!")

    # Distribution clsloc 2 det categories
    det_num_clsloc = {}
    for catid in syn_data.det_synsets_id.keys(): det_num_clsloc[catid] = 0
    for wnid, parentid in syn_data.clsloc_2_det.items():
        catid = syn_data.det_synsets[parentid]["id"]
        det_num_clsloc[catid] += 1

    # Images in each category
    print("Computing image distribution in each of the 200 categories...")
    # Create categories first
    det_categories = {}
    num_assigned_img = 0
    overall_chance_top1 = 0
    overall_chance_top5 = 0
    for idx in sorted(syn_data.det_synsets_id.keys()):
        syn = syn_data.det_synsets[syn_data.det_synsets_id[idx]]
        wnid, name, desc = [syn[att] for att in ["wnid", "name", "desc"]]
        cat_idx_mask = labels == idx
        cat_idx = np.array(np.where(cat_idx_mask)[0])
        cat_num = len(cat_idx)
        num_of_clsloc = det_num_clsloc[idx]
        random_chance_top1 = det_prob(num_of_clsloc, 1)
        random_chance_top5 = det_prob(num_of_clsloc, 5)
        overall_chance_top1 += random_chance_top1*cat_num
        overall_chance_top5 += random_chance_top5 * cat_num
        # Populate image indexes
        det_cat = DetCategory(data_single=val_data_single, idx=idx, wnid=wnid, name=name, desc=desc,
                              indexes=cat_idx, num=cat_num, index_mask=cat_idx_mask,
                              num_clsloc_cats=num_of_clsloc,
                              random_chance_top1=random_chance_top1,
                              random_chance_top5=random_chance_top5
                              )
        det_categories[idx] = det_cat
        num_assigned_img += cat_num

    overall_chance_top1 = overall_chance_top1 / num_single
    overall_chance_top5 = overall_chance_top5 / num_single

    assert(num_assigned_img == num_single)

    print("Done!")

    return ValDatabase(val_data, val_data_single, single_idx, num_single, labels,
                       det_categories=det_categories,
                       random_chance_top1=overall_chance_top1,
                       random_chance_top5=overall_chance_top5)

def load_val_data():
    print("Loading validation data...")
    with open(val_pickle_file,'rb') as f:
        # Deal with python2 -> 3 mess
        val_data = pickle.load(f)

    print("Data loaded! {} images".format(len(val_data)))
    return val_data

def generate_val_data():
    syn_data = load_syn_data()
    with open(val_idx_path) as f:
        val_idx_str = f.read().split("\n")
    # Read index -> val image file
    val_idx = {}
    for line in val_idx_str:
        line = line.strip()
        if line:
            v_name, v_idx = line.split(" ")
            val_idx[int(v_idx)] = v_name

    val_data = {}
    for idx in sorted(val_idx.keys()):
        val_data[idx] = read_val_metadata(idx, val_idx, syn_data)

    return val_data

def save_val_data():
    print("Generating val data..")
    val_data = generate_val_data()
    print("Saving val data to pickle: {}".format(val_pickle_file))
    with open(val_pickle_file, 'wb') as f:
        pickle.dump(val_data, f)

def read_val_metadata(idx, val_idx, syn_data):
    data_xml = val_annot_folder + val_idx[idx] + ".xml"
    tree = ET.parse(data_xml)
    root = tree.getroot()
    filename = root.findall('filename')[0].text
    img_path = val_img_folder + filename + ".JPEG"
    size = root.findall('size')[0]
    width = int(size.findall('width')[0].text)
    height = int(size.findall('height')[0].text)
    all_obj = root.findall('object')
    objects = []
    for ob in all_obj:
        wnid = ob.findall('name')[0].text
        bndbox = ob.findall('bndbox')[0]
        xmin, xmax, ymin, ymax = [int(bndbox.findall(tag)[0].text) for tag in ['xmin','xmax','ymin','ymax']]
        objects.append(ObjectAnnotation(wnid=wnid, synset=syn_data.det_synsets[wnid],
                                       xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax))
    return ValImage(idx=idx, img_path=img_path, objects=objects, width=width, height=height)


def generate_one_object_database(val_data):
    print("Finding val images with only one object..")
    val_data_single = []
    single_idx = []
    for i, v in enumerate(val_data.values()):
        if len(v.objects) == 1:
            val_data_single.append(v)
            single_idx.append(v.idx)
    num_single = len(single_idx)
    print("One object database built: {} images".format(num_single))
    return val_data_single, single_idx, num_single



if __name__ == "__main__":
    save_val_data()
    print("Testing loading val data from pickle")
    db = load_val_database()
