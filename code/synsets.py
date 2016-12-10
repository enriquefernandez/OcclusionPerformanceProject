import scipy.io as sio
import attr
import json
import numpy as np


@attr.s
class SynData(object):
    clsloc_synsets = attr.ib()
    clsloc_synsets_id = attr.ib()
    det_synsets = attr.ib()
    det_synsets_id = attr.ib()
    clsloc_2_det = attr.ib()
    keras_imagenet_idx = attr.ib()

    def decode_predictions_det(self, clsloc_pred):
        all_preds = []
        for img in clsloc_pred:
            preds = []
            for el in img:
                if el[0] in self.clsloc_2_det:
                    wnid = self.clsloc_2_det[el[0]]
                    syn = self.det_synsets[wnid]
                    name = syn['name']
                    preds.append((wnid, name, el[2]))
                else:
                    preds.append((None, None, el[2]))
            all_preds.append(preds)
        return all_preds

    def decode_det_category(self, clsloc_cats):
        det_cats = []
        for cat in clsloc_cats:
            wnid = self.clsloc_synsets_id[cat]
            if wnid in self.clsloc_2_det:
                wnid_det = self.clsloc_2_det[wnid]
                det_cats.append(self.det_synsets[wnid_det]["id"])
        return det_cats

    def clsloc2det_cat_matrix(self, clsloc_cat_matrix):
        det_mat = np.empty(clsloc_cat_matrix.shape)
        v_conversion = np.vectorize(lambda x: self.clsloc_cat_2_det_cat(x))
        return v_conversion(clsloc_cat_matrix)

    def clsloc_cat_2_det_cat(self, clsloc_cat):
        """clsloc_cat is the category index.
        Returns det category index or NaN if no conversion available."""
        wnid = self.clsloc_synsets_id[clsloc_cat]
        if wnid in self.clsloc_2_det:
            wnid_det = self.clsloc_2_det[wnid]
            return self.det_synsets[wnid_det]["id"]
        else:
            return 0 #0 is not a valid DET category, so consider this failure to convert

    def det_name_from_id(self, idx):
        return str(self.det_synsets[self.det_synsets_id[idx]]["name"])


def det_clsloc_names(syn_data, det_id, clsloc_id):
    if det_id > 0:
        det_syn = syn_data.det_synsets[syn_data.det_synsets_id[det_id]]["name"]
        det_syn_name = "{} (id: {})".format(det_syn, det_id)
    else:
        det_syn_name = "-"
    clsloc_syn = syn_data.clsloc_synsets[syn_data.clsloc_synsets_id[clsloc_id]]["name"]
    clsloc_syn_name = "{} (id: {})".format(clsloc_syn, clsloc_id)
    return det_syn_name, clsloc_syn_name


def load_syn_data():
    # Det synsets
    data = sio.loadmat('../ILSVRC2015/devkit/data/meta_det.mat')

    det_synsets = {}
    det_synsets_id = {}

    for i, d in enumerate(data['synsets'][0][:]):
        sid = int(d[0].flatten()[0])
        wnid = d[1][0]
        name = d[2][0]
        desc = d[3][0] if len(d[3]) > 0 else ""
        det_synsets[wnid] = {"id": sid, "wnid": wnid, "name": name, "desc": desc}
        det_synsets_id[sid] = wnid
    print("Loaded DET data")

    # clsloc synsets
    data = sio.loadmat('../ILSVRC2015/devkit/data/meta_clsloc.mat')
    clsloc_synsets = {}
    clsloc_synsets_id = {}

    for i, d in enumerate(data['synsets'][0][:]):
        sid = int(d[0].flatten()[0])
        wnid = d[1][0]
        name = d[2][0]
        desc = d[3][0] if len(d[3]) > 0 else ""
        clsloc_synsets[wnid] = {"id": sid, "wnid": wnid, "name": name, "desc": desc}
        clsloc_synsets_id[sid] = wnid
    print("Loaded CLSLOC data")

    # Relation between CLSLOC and DET
    # CLSLOC (1000 categories) to DET (200 categories)

    # Load parent child relation file
    with open('../wordnet/wordnet_parent_child.txt') as f:
        parent_child = f.read()

    parent_child = parent_child.split('\n')

    is_a = {}

    for line in parent_child[:]:
        if line:
            parent, child = line.split(" ")
            if child in is_a:
                pass
            is_a[child] = parent

    notfound = 0
    clsloc_2_det = {}
    clsloc_2_det_id = {}

    for sid in range(1, 1001):
        clsloc_wnid = clsloc_synsets_id[sid]
        synset = clsloc_synsets[clsloc_wnid]

        found = False
        parent = clsloc_wnid

        trace = [parent]

        while not found:
            if parent in det_synsets:
                found = True
                parent_syn = det_synsets[parent]
                #             print "Found!"
                # print("Synset id:{},{},{} is a id:{}, {}, {}".
                #       format(synset['id'], synset['wnid'], synset['name'],
                #              parent_syn['id'], parent_syn['wnid'], parent_syn['name']))
                clsloc_2_det[clsloc_wnid] = parent
                clsloc_2_det_id[synset['id']] = parent_syn['id']
            else:
                if parent in is_a:
                    parent = is_a[parent]
                    trace.append(parent)
                else:
                    notfound += 1
                    # print("------------- COULDN't figure out {}, {}, {}".
                    #       format(synset['id'], synset['wnid'], synset['name']))
                    # print("------Last parent is: {}. Trace: {}".format(parent, trace))
                    break
        if not found:
            pass
    print("Computed relation between CLSLOC and DET.")

    print("Loading Keras imagenet class index")
    # Keras imagenet clsloc index
    keras_imagenet_idx = json.load(open("imagenet_class_index.json"))

    return SynData(clsloc_synsets, clsloc_synsets_id, det_synsets, det_synsets_id, clsloc_2_det,
                   keras_imagenet_idx)
