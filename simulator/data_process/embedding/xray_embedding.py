# coding: utf-8
import sys
from project_path import pro_path
sys.path.append(pro_path)
import sys

sys.path.insert(0, "../../..")
import skimage.io
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import cv2
import numpy as np
from tqdm import tqdm
import os


#  Embedded chest images, orig_imgs_weights are weights based on distance from examination time to admission time by hour 1D list
def multi_chest_xray_embeddings(imgs, orig_imgs_weights):
    # sys.stdout.flush()
    densefeature_embeddings = []
    prediction_embeddings = []
    root = f'{pro_path}datasets/'#jpg_files path
    cuda_available = torch.cuda.is_available()

    nImgs = len(imgs)
    with tqdm(total=nImgs) as pbar:
        for idx, img_path in enumerate(imgs):
            img_path = root + img_path

            if not os.path.exists(img_path):
                return None, None
            img = skimage.io.imread(img_path)
            img = xrv.datasets.normalize(img, 255)

            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("Error: Dimension lower than 2 for image!")

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = img[None, :, :]
            model = xrv.models.DenseNet(weights='densenet121-res224-chex')

            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                if cuda_available:
                    img = img.cuda()
                    model = model.cuda()

                # Extract dense features
                feats = model.features(img)
                feats = F.relu(feats, inplace=True)
                feats = F.adaptive_avg_pool2d(feats, (1, 1))
                densefeatures = feats.cpu().detach().numpy().reshape(-1)
                densefeature_embeddings.append(densefeatures)  # append to list of dense features for all images

                preds = model(img).cpu()
                predictions = preds[0].detach().numpy()
                prediction_embeddings.append(predictions)  # append to list of predictions for all images

    orig_imgs_weights = np.asarray(orig_imgs_weights)
    adj_imgs_weights = orig_imgs_weights - orig_imgs_weights.min()
    imgs_weights = (adj_imgs_weights) / (adj_imgs_weights).max()

    try:
        aggregated_densefeature_embeddings = np.average(densefeature_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_densefeature_embeddings)):
            aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
    except:
        aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])

    try:
        aggregated_prediction_embeddings = np.average(prediction_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_prediction_embeddings)):
            aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
    except:
        aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])

    return aggregated_densefeature_embeddings, aggregated_prediction_embeddings


def single_chest_xray_embeddings(img_path):
    # sys.stdout.flush()
    cuda_available = torch.cuda.is_available()

    root = f'{pro_path}datasets/'#jpg_files path
    img_path = root + img_path

    if not os.path.exists(img_path):
        print(f'Please check that the cxr jpg file is missing!({img_path})')
        return None, None
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)

    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("Error: Dimension lower than 2 for image!")

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img[None, :, :]
    model = xrv.models.DenseNet(weights='densenet121-res224-chex')

    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda_available:
            img = img.cuda()
            model = model.cuda()

        # Extract dense features
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        densefeatures = feats.cpu().detach().numpy().reshape(-1)
        densefeature_embeddings = densefeatures

        preds = model(img).cpu()
        predictions = preds[0].detach().numpy()
        prediction_embeddings = predictions

    return densefeature_embeddings, prediction_embeddings

