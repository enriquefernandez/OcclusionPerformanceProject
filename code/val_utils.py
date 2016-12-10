from matplotlib import pyplot as plt
import numpy as np
import cv2
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions


def get_val_img(val_data, idx, size=None):
    val = val_data[idx]
    if size == None:
        size = (val.height, val.width)
    img_path = val.img_path
    img = image.load_img(img_path, target_size=size)
    return img, val

# def get_db_image(val, size=None):
#     # val = val_data[idx]
#     if size == None:
#         size = (val.height, val.width)
#     img_path = val.img_path
#     img = image.load_img(img_path, target_size=size)
#     return img

def get_db_image(val):
    img_path = val.img_path
    img = cv2.imread(img_path)
    return img


def display_val(idx):
    img, val = get_val_img(idx)
    print("width: {}, height: {}".format(val.width, val.height))
    for i, ob in enumerate(val.objects):
        print("Object {}: {}, {}".format(i, ob.wnid, ob.synset["name"]))
    return img


def display_cv(val_data, idx):
    val = val_data[idx]
    img_path = val.img_path
    img = cv2.imread(img_path)
    for i, ob in enumerate(val.objects):
        print("Object {}: {}, {}".format(i, ob.wnid, ob.synset["name"]))
        minx, miny, maxx, maxy = ob.xmin, ob.ymin, ob.xmax, ob.ymax
        cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
    plt.imshow(img)


def classify_img(model, syn_data, img):
    # Resize image
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # val_truth = det_synsets[label]
    clsloc_preds = decode_predictions(preds)
    det_preds = syn_data.decode_predictions_det(clsloc_preds)
    # print('Validation label: {},{},{}'.format(val_truth['id'],val_truth['wnid'],val_truth['name']))
    #     print('Predicted clsloc:', clsloc_preds)
    #     print('Predicted det:', det_preds)
    return preds, det_preds, clsloc_preds


def classify_val(model, syn_data, idx):
    img, val = get_val_img(idx, size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # val_truth = det_synsets[label]
    clsloc_preds = decode_predictions(preds)
    det_preds = syn_data.decode_predictions_det(clsloc_preds)
    # print('Validation label: {},{},{}'.format(val_truth['id'],val_truth['wnid'],val_truth['name']))
    print('Predicted clsloc:', clsloc_preds)
    print('Predicted det:', det_preds)
    return preds, det_preds, clsloc_preds


def nice_preds(res):
    det = res[1]
    all_nice_pred = []
    for img in det:
        nice_pred = []
        for p in img:
            if p[0]:
                nice_pred.append("{} ({})".format(p[1], p[0]))
        all_nice_pred.append(nice_pred)

    clsloc = res[2]
    all_nice_cslsoc = []
    for img in clsloc:
        nice_cslsoc = []
        for p in img:
            nice_cslsoc.append("{} ({})".format(p[1], p[0]))
        all_nice_cslsoc.append(nice_cslsoc)
    return zip(all_nice_pred, all_nice_cslsoc)