from __future__ import division
from models.resnet50 import ResNet50
from models.vgg19 import VGG19
from models.vgg16 import VGG16
from models.inception_v3 import InceptionV3, preprocess_input as preprocess_input_inception
from keras.optimizers import SGD
from convnetskeras.convnets import convnet
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import h5py
import os

from synsets import load_syn_data
from val_data import load_val_database, ValImage, ObjectAnnotation
from data_loader import ValDataLoader

results = "../results/"
bb_folder = "../bounding_boxes/"

# batch_size = 64 # needed for inception
batch_size = 32 # needed for vgg19

def alexnet_preprocess(x):
    "x will be (1, 3, 227, 227)"
    # We normalize the colors (in RGB space) with the empirical means on the training set
    img[:, 0, :, :] -= 123.68
    img[:, 1, :, :] -= 116.779
    img[:, 2, :, :] -= 103.939





def compute_accuracy(labels, topk):
    # cats_mat = top5_det[:, :5]
    res_mat = np.zeros(topk.shape, dtype=np.bool)
    num_c = topk.shape[1]
    for c in range(num_c):
        res_mat[:, c] = labels == topk[:, c]
    result = np.sum(res_mat, axis=1, dtype=np.bool)
    return result


def predict(model, syn_data, data_loader, batch_size, out_file):

    data_loader.reset()

    num_batches = data_loader.num_batches(batch_size)
    num_single = data_loader.num

    overall_t = time.time()
    img_processed = 0
    h5_name = os.path.join(results, out_file)

    f_h5 = h5py.File(h5_name, "w")
    print("Opened {} h5 file".format(h5_name))

    # pred_dataset = f_h5.create_dataset("predictions", (num_single, 1000), dtype='float64')
    pred_dataset = f_h5.create_dataset("top5_preds", (num_single, 5), dtype='float64')
    top5clsloc_dataset = f_h5.create_dataset("top5_clsloc", (num_single, 5), dtype='i')

    for i in range(num_batches):
        start_t = time.time()
        print("Preproccessing batch ({}/{})...".format(i + 1, num_batches))
        batch = data_loader.prepare_batch(batch_size)
        num_img = len(batch)
        print("\t\tPreproccessed {} img in {:.2f} seconds.".format(num_img, time.time() - start_t))
        print("{}: Predicting batch ({}/{}) of size {}...".format(out_file, i + 1, num_batches, num_img))
        start_t = time.time()
        preds = model.predict_on_batch(batch)

        for j, img_pred in enumerate(preds):
            best_5 = img_pred.argsort()[::-1][:5]  # In Keras id numbers, not mine
            best_5 = [syn_data.keras_imagenet_idx[str(idx)][0] for idx in best_5]  # wnids
            best_5 = [syn_data.clsloc_synsets[c]["id"] for c in best_5] # clsloc cat id numbers
            #         print("Best 5: ", best_5)
            top5clsloc_dataset[img_processed + j, ...] = best_5

            best_5_preds = np.sort(img_pred)[::-1][:5]  # Top 5 values
            sum_preds = np.sum(img_pred)
            prop_best5_preds = best_5_preds / sum_preds
            # print("Best5:", best_5_preds)
            # print("Sum:", sum_preds)
            # print("Prop: ", prop_best5_preds)
            pred_dataset[img_processed + j, ...] = prop_best5_preds

        # pred_dataset[img_processed:img_processed + num_img, ...] = preds

        print("\t\tDone!. Predicted in {:.2f} seconds".format(time.time() - start_t))
        img_processed += num_img

    # Convert clsloc cat ids to DET cat ids
    print("Converting clsloc cat ids to DET cat ids...")
    top5_categories = syn_data.clsloc2det_cat_matrix(top5clsloc_dataset[:,:])
    top5det_dataset = f_h5.create_dataset("top5_det", (num_single, 5), dtype='i',
                                          data=top5_categories)
    # Compute top1 and top5 accuracy
    print("Computing top1 and top5 accuracy")
    top5_ok = f_h5.create_dataset("correct_top1", (num_single, 1), dtype='bool',
                                  data=compute_accuracy(db.labels, top5_categories[:, :1]))
    top5_ok = f_h5.create_dataset("correct_top5", (num_single, 1), dtype='bool',
                                             data=compute_accuracy(data_loader.db.labels, top5_categories[:,:5]))

    print("\nFinishing predicting all {} images in {:.2f} seconds".format(img_processed, time.time() - overall_t))
    f_h5.close()
    pass


def load_model(model_name):
    print("Loading {} model...".format(model_name))
    if model_name == "resnet50":
        model =  ResNet50(weights='imagenet')
    elif model_name == "inceptionv3":
        model = InceptionV3(weights='imagenet')
    elif model_name == "vgg19":
        model = VGG19(weights='imagenet')
    elif model_name == "vgg16":
        model = VGG16(weights='imagenet')
    elif model_name == "alexnet":
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model = convnet('alexnet', weights_path="../alexnet/alexnet_weights.h5", heatmap=False)
        model.compile(optimizer=sgd, loss='mse')
    else:
        raise ValueError("Wrong model name")
    print("Loaded!")
    return model, model_name

def predict_with_model(model, model_name, data_name, data_loader, batch_size=50):
    out_file = "preds_{}_{}.h5".format(model_name, data_name)
    print("\n\nABOUT TO GENERATE PREDICTIONS FOR ORIGINAL IMAGES WITH {}____\n".
          format(model_name))
    predict(model, syn_data, data_loader, batch_size=batch_size, out_file=out_file)
    print("\nDONE GENERATING PREDICTIONS FOR ORIGINAL DATA AND {}!\n".
          format(model_name))
def free_model(model, model_name):
    print("Freeing model {}...".format(model_name))
    del model
    print("Done.")


def predict_center_occlusion(model, db, size, batch_size, img_order='tf', preproc_fun=None):
    # Center occlusions
    data_names = ["bb_center_010", "bb_center_020", "bb_center_050", "bb_center_075", "bb_center_100"]
    for data_name in data_names:
        bb_db_file = os.path.join(bb_folder, data_name) + ".h5"
        data_loader = ValDataLoader(db, bb_db_file=bb_db_file, target_size=size, img_order=img_order,
                                    preproc_fun=preproc_fun)
        predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)

def predict_corner_occlusion(model, db, size, batch_size, img_order='tf', preproc_fun=None):
    # Corner occlusions
    data_names = ["bb_corner_010", "bb_corner_020", "bb_corner_050", "bb_corner_075"]
    for data_name in data_names:
        bb_db_file = os.path.join(bb_folder, data_name) + ".h5"
        data_loader = ValDataLoader(db, bb_db_file=bb_db_file, target_size=size, img_order=img_order,
                                    preproc_fun=preproc_fun)
        predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)

def predict_random_occlusion(model, db, size, batch_size, img_order='tf', preproc_fun=None):
    # Corner occlusions
    data_names = ["bb_random_010", "bb_random_020", "bb_random_050", "bb_random_075"]
    for data_name in data_names:
        bb_db_file = os.path.join(bb_folder, data_name) + ".h5"
        data_loader = ValDataLoader(db, bb_db_file=bb_db_file, target_size=size, img_order=img_order,
                                    preproc_fun=preproc_fun)
        predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)


if __name__ == "__main__":
    overall_start = time.time()
    syn_data = load_syn_data()
    db = load_val_database(syn_data)


    # # # Resnet50
    # # # Original data
    # model, model_name = load_model('resnet50')
    # size, batch_size = (224, 224), 64
    # data_name = "original"
    # data_loader = ValDataLoader(db, target_size=size)
    # predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)
    # predict_center_occlusion(model,db, size, batch_size)
    # predict_corner_occlusion(model, db, size, batch_size)
    # predict_random_occlusion(model, db, size, batch_size)
    # free_model(model, model_name)
    # #
    # Inception v3
    # Original data
    model, model_name = load_model('inceptionv3')
    size, batch_size = (299, 299), 32
    data_name = "original"
    data_loader = ValDataLoader(db, target_size=size, preproc_fun=preprocess_input_inception)
    # predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)
    predict_center_occlusion(model,db, size, batch_size, preproc_fun=preprocess_input_inception)
    predict_corner_occlusion(model, db, size, batch_size, preproc_fun=preprocess_input_inception)
    predict_random_occlusion(model, db, size, batch_size, preproc_fun=preprocess_input_inception)
    free_model(model, model_name)
    #
    # # # VGG19
    # # # Original data
    # model, model_name = load_model('vgg19')
    # size, batch_size = (224, 224), 32
    # data_name = "original"
    # data_loader = ValDataLoader(db, target_size=size)
    # predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)
    # predict_center_occlusion(model,db, size, batch_size)
    # predict_corner_occlusion(model, db, size, batch_size)
    # predict_random_occlusion(model, db, size, batch_size)
    # free_model(model, model_name)
    #
    # # # VGG16
    # # # Original data
    # model, model_name = load_model('vgg16')
    # size, batch_size = (224, 224), 32
    # data_name = "original"
    # data_loader = ValDataLoader(db, target_size=size)
    # predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)
    # predict_center_occlusion(model,db, size, batch_size)
    # predict_corner_occlusion(model, db, size, batch_size)
    # predict_random_occlusion(model, db, size, batch_size)
    # free_model(model, model_name)
    #
    # # AlexNet
    # # Original data
    # model, model_name = load_model('alexnet')
    # size, batch_size = (227, 227), 64
    # data_name = "original"
    # data_loader = ValDataLoader(db, target_size=size,
    #                             img_order='th')
    # predict_with_model(model, model_name, data_name, data_loader, batch_size=batch_size)
    # predict_center_occlusion(model,db, size, batch_size, img_order='th')
    # predict_corner_occlusion(model, db, size, batch_size, img_order='th')
    # predict_random_occlusion(model, db, size, batch_size, img_order='th')
    # free_model(model, model_name)

    print("\n\nFinished all in {:.2f} seconds".format(time.time()-overall_start))

    # Finishes everything in 9788.21 seconds

