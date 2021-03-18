# +
import train
import torch
from radionets.evaluation.utils import  load_pretrained_model, eval_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import model
from model import center_to_boundary
from radionets.dl_framework.data import get_bundles
from tqdm import tqdm

def box_coord(coord, img_size):
    x = coord[0].item()*img_size
    y = coord[3].item()*img_size
    xmax = coord[2]
    ymin = coord[1]
    w = xmax.item()*img_size - x
    h = -(y - ymin.item()*img_size)
    return x,y,w,h

def detect_sources(checkpoint_path, data_path, img_size, n = 0):
    data = get_bundles(data_path)
    eval_dataset = train.detect_dataset(data)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = 32,
                                              shuffle = False,
                                              collate_fn = eval_dataset.collate_fn)
    checkpoint = checkpoint_path
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to('cuda')
    model.eval()
    #print(eval_dataset[31])
    with torch.no_grad():
        for i, (images, boxes, labels) in enumerate(tqdm(eval_loader)):
            print(enumerate(eval_loader))
            images = images.to('cuda')
            print(images.shape)
            predicted_locs, predicted_scores = model(images)
            predb, predl, preds = model.object_detection(predicted_locs, predicted_scores,priors= model.priors_cxcy,
                                     min_score = 0.2, max_overlap = 0.45, top_k = 10)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,8))
    for j in range(len(eval_dataset[n][1][0])):
        trux, truy, truw, truh = box_coord(eval_dataset[n][1][0][j],img_size)
        trurect = patches.Rectangle((trux, truy), truw, truh, linewidth=1, edgecolor='g', facecolor='none', label = eval_dataset[n][2][0][j].item())
        ax1.add_patch(trurect)
    print(preds[n])
    for k in range(len(predl[n])):
        predx, predy, predw, predh = box_coord(predb[n][k],img_size)
        predrect = patches.Rectangle((predx, predy), predw, predh, linewidth=1, edgecolor='r', facecolor='none', label = predl[n][k].item())
        ax2.add_patch(predrect)
        
    ax1.imshow(eval_dataset[n][0].squeeze(0))
    ax2.imshow(eval_dataset[n][0].squeeze(0))
    ax1.legend()
    ax2.legend()
    
def image_detection(checkpoint, image):
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        image = image.to('cuda')
        predicted_locs, predicted_scores = model(image)
        predb, predl, preds = model.object_detection(predicted_locs, predicted_scores,priors= model.priors_cxcy,
                                     min_score = 0.2, max_overlap = 0.1, top_k = 100)
    return predb, predl

def classifier_eval(arch, img_batch):
    
    pred = eval_model(img, arch)
    _, l = torch.max(pred, dim = 1)
    return l
