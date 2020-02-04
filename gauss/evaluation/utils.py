import torch
from skimage.feature import blob_log

def num_blobs(img):
    img = img.view(64,64)
    blobs = blob_log(img,min_sigma=0.1, max_sigma=7, num_sigma=30, threshold=0.3, overlap=0.7)
    return blobs.shape[0]
