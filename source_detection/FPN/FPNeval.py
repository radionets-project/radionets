# +
import FPNtrain
import torch
from radionets.evaluation.utils import  load_pretrained_model, eval_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import FPN
from FPN import center_to_boundary
from radionets.dl_framework.data import get_bundles
from tqdm import tqdm
import matplotlib.patches as mpatches

class_labels = ('pointlike gaussian', 'diffuse gaussian', 'diamond', 'square', 'background')
color_map = ('w', 'r', '#84b819', 'cyan','brown')
label_map = {k: v for v, k in enumerate(class_labels)}
rev_label_map = {v: k for k, v in label_map.items()} 
def box_coord(coord, img_size):
    x = coord[0].item()*img_size #0
    y = coord[3].item()*img_size#3
    xmax = coord[2]#2
    ymin = coord[1]#1
    w = xmax.item()*img_size - x
    h = -(y - ymin.item()*img_size)
    return x,y,w,h
def box_coord_inv(coord):
    newcoord = np.zeros(coord.shape)
    xmin = coord[:,0]
    ymax = coord[:,1]
    xmax = xmin+coord[:,2]
    ymin = ymax + coord[:,3]
    newcoord[:,0] = xmin
    newcoord[:,1] = ymin
    newcoord[:,2] = xmax
    newcoord[:,3] = ymax
    return newcoord

def detect_sources(checkpoint_path, data_path, img_size, n = 0):
    data = get_bundles(data_path)
    eval_dataset = FPNtrain.detect_dataset(data)
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
            images = images.to('cuda')
            predicted_locs, predicted_scores = model(images)
            predb, predl, preds = model.object_detection(predicted_locs, predicted_scores,priors= model.priors_cxcy,
                                     min_score = 0.3, max_overlap = 0.45, top_k = 100)
    plt.rcParams.update({'font.size': 20})
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,8))
   
    for j in range(len(eval_dataset[n][1][0])):
        true_label = class_labels[eval_dataset[n][2][0][j].item()]
        color = color_map[eval_dataset[n][2][0][j].item()]
        trux, truy, truw, truh = box_coord(eval_dataset[n][1][0][j],img_size)
        trurect = patches.Rectangle((trux, truy), truw, truh, linewidth=2, edgecolor=color, facecolor='none')
        #ax1.text(trux,(truy+truh-7),true_label, color = 'k',fontsize=8,backgroundcolor = color)
        ax1.add_patch(trurect)
    
    for k in range(len(predl[n])):
        predicted_label = class_labels[predl[n][k].item()]
        color = color_map[predl[n][k].item()]
        predx, predy, predw, predh = box_coord(predb[n][k],img_size)
        predrect = patches.Rectangle((predx, predy), predw, predh, linewidth=2, edgecolor=color,
                                     facecolor='none')
        #ax2.text(predx,(predy+predh-7),predicted_label, color = 'k',fontsize=8,backgroundcolor = color)
        ax2.add_patch(predrect)
    
    cbar_ax = fig.add_axes([0.982, 0.183, 0.03, 0.632])
    
    ax1.set(xlabel='Pixels', ylabel='Pixels')
    ax1.label_outer()
    ax2.set(xlabel='Pixels', ylabel='Pixels')
    ax2.label_outer()
    
    #ax1.set(xlabel='', ylabel='Pixels')
    #ax1.label_outer()
    #ax2.set(xlabel='', ylabel='Pixels')
    #ax2.label_outer()
    #ax1.set_xticks([], [])
    #ax2.set_xticks([], [])
    
    print('max', eval_dataset[n][0].squeeze(0).max())
    
    img2 = ax1.imshow(eval_dataset[n][0].squeeze(0),cmap = 'gist_heat')
    img = ax2.imshow(eval_dataset[n][0].squeeze(0),cmap = 'gist_heat')
    cbar = fig.colorbar(img, cax=cbar_ax,label = 'Normalized Intensity')
    
    green_patch = mpatches.Patch(color='cyan', label='Square')
    white_patch = mpatches.Patch(facecolor='w',edgecolor='black', label='Pointlike Gaussian')
    red_patch = mpatches.Patch(color='red', label='Diffuse Gaussian')
    pink_patch = mpatches.Patch(color='#84b819', label='diamond')
    #plt.figlegend([white_patch,red_patch, pink_patch, green_patch],['Pointlike Gaussian','Diffuse Gaussian', 'Diamond', 'Square'],loc =9,ncol=4, bbox_to_anchor=(0.1, .45, .9, 0.5))
    fig.tight_layout()
    return fig
def image_detection(checkpoint_path, image):
    img_size = image.shape[0]
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    checkpoint = checkpoint_path
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to('cuda')
    model.eval()
    priors = FPN.create_prior_boxes()
    with torch.no_grad():
        image = image.to('cuda')
        predicted_locs, predicted_scores = model(image)

        predb, predl, preds = model.object_detection(predicted_locs, predicted_scores, model.priors_cxcy,
                                     min_score = 0.3, max_overlap = 0.45, top_k = 100)

    return predb, predl,preds

def classifier_eval(arch, img_batch):
    
    pred = eval_model(img, arch)
    _, l = torch.max(pred, dim = 1)
    return l

def mAPeval(checkpoint_path, data_path, curve = False):
    data = get_bundles(data_path)
    eval_dataset = FPNtrain.detect_dataset(data)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = 32,
                                              shuffle = False,
                                              collate_fn = eval_dataset.collate_fn)
    checkpoint = checkpoint_path
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to('cuda')
    model.eval()
    pred_boxes = list()
    pred_labels = list()
    pred_scores = list()
    true_boxes = list()
    true_labels = list()
    
    with torch.no_grad():
        for i, (images, boxes, labels) in enumerate(tqdm(eval_loader)):
            images = images.to('cuda')
           # boxes = boxes.to('cuda')
           # scores = scores.to('cuda')
            predicted_locs, predicted_scores = model(images)
            predb, predl, preds = model.object_detection(predicted_locs, predicted_scores,priors= model.priors_cxcy,
                                     min_score = 0.3, max_overlap = 0.45, top_k = 100)
            boxes = [boxes[b][0] for b in range(len(boxes))]
            labels = [labels[l][0][0] for l in range(len(labels))]
            pred_boxes.extend(predb)
            pred_labels.extend(predl)
            pred_scores.extend(preds)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
        
        if curve:
             c = calculate_mAP(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, curve)
             return c
        else:
            APs, mAP = calculate_mAP(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, curve)
            print(APs)
            print('\nMean Average Precision: %.3f' %mAP)

def calculate_mAP(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, curve):
        assert len(pred_boxes) == len(pred_labels) == len(pred_scores) == len(true_boxes) == len(
        true_labels)
        
        n_classes = len(label_map)
        
        true_images = list()
        for i in range(len(true_labels)):
            true_images.extend([i] * len(true_labels[i]))
        true_images = torch.LongTensor(true_images).to('cuda')  
        true_boxes = torch.cat(true_boxes, dim=0)  
        true_labels = torch.cat(true_labels, dim=0)  
        
        pred_images = list()
        for i in range(len(pred_labels)):
            pred_images.extend([i] * pred_labels[i].size(0))
        pred_images = torch.LongTensor(pred_images).to('cuda')
        pred_boxes = torch.cat(pred_boxes, dim=0) 
        pred_labels = torch.cat(pred_labels, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)
        
        average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)
        curve_values = []
        #BEWARE BELOW
        print(n_classes-1)
        for c in range(0, n_classes-1):
            true_class_images = true_images[true_labels == c] 
            true_class_boxes = true_boxes[true_labels == c] 
            
            
            true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(
                'cuda') 

            
            pred_class_images = pred_images[pred_labels == c]  
            pred_class_boxes = pred_boxes[pred_labels == c]  
            pred_class_scores = pred_scores[pred_labels == c] 
            n_class_detections = pred_class_boxes.size(0)
            if n_class_detections == 0:
                continue

           
            pred_class_scores, sort_ind = torch.sort(pred_class_scores, dim=0, descending=True)  
            pred_class_images = pred_class_images[sort_ind]  
            pred_class_boxes = pred_class_boxes[sort_ind]  
            
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to('cuda') 
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to('cuda')  
            for d in range(n_class_detections):
                this_detection_box = pred_class_boxes[d].unsqueeze(0) 
                this_image = pred_class_images[d] 

                
                object_boxes = true_class_boxes[true_class_images == this_image] 
                
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                
                overlaps = FPN.jaccard(this_detection_box, object_boxes)  
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  
                
                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
               
                
                if max_overlap.item() > 0.5:
                     if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1 
                       
                     else:
                        false_positives[d] = 1
                
                else:
                    false_positives[d] = 1

            
            cumul_true_positives = torch.cumsum(true_positives, dim=0)  
            cumul_false_positives = torch.cumsum(false_positives, dim=0) 
            cumul_precision = cumul_true_positives / (
                    cumul_true_positives + cumul_false_positives + 1e-10) 
            cumul_recall = cumul_true_positives / len(true_class_images)  
            if curve:
                c_recall_thresholds = torch.arange(start=0, end=1.1, step=.01).tolist()  # (11)
                c_precisions = torch.zeros((len(c_recall_thresholds)), dtype=torch.float).to('cuda')  # (11)
                for i, t in enumerate(c_recall_thresholds):
                    c_recalls_above_t = cumul_recall >= t
                    if c_recalls_above_t.any():
                        c_precisions[i] = cumul_precision[c_recalls_above_t].max()
                    else:
                        c_precisions[i] = 0.
                curve_values.append(c_precisions)


            recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to('cuda')  # (11)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if recalls_above_t.any():
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.
            average_precisions[c] = precisions.mean()  # c is in [1, n_classes - 1]
    

        if curve == False:
            mean_average_precision = average_precisions.mean().item()


            average_precisions = {rev_label_map[c]: v for c, v in enumerate(average_precisions.tolist())}

            return average_precisions, mean_average_precision
        else:
            return curve_values



# -
def open_bundle_pack(path):
    bundle_x = []
    bundle_y = []
    bundle_z = []
    f = h5py.File(path, "r")
    bundle_size = len(f)//3
    for i in range(bundle_size):
        bundle_x_i = np.array(f["x"+str(i)])
        bundle_y_i = np.array(f["y"+str(i)])
        bundle_z_i = np.array(f["z"+str(i)])
        bundle_x.append(bundle_x_i)
        bundle_y.append(bundle_y_i)
        bundle_z.append(bundle_z_i)
    return bundle_x, bundle_y, bundle_z


def annotate(img, bbox, labels):
    #class_labels = ('pointlike gaussian', 'diffuse gaussian', 'diamond', 'square', 'background')
    #color_map = ('w', 'g', 'r', 'y','brown')
    img_size = img.shape[0]
    fig, ax2 = plt.subplots(1,1,figsize=(50,40))
    for j in range(bbox.shape[0]):
        true_label = labels[j]
        color = color_map[labels[j].astype('int')]
        trux, truy, truw, truh = box_coord(bbox[j],img_size)
        trurect = patches.Rectangle((trux, truy), truw, truh, linewidth=2, edgecolor=color, facecolor='none')
        #ax2.text(trux,(truy+truh-7),true_label, color = 'k',fontsize=8,backgroundcolor = color)
        ax2.add_patch(trurect)
    mi = ax2.imshow(img, cmap = 'gist_heat')
    ax2.axis('off')
    cbar_ax = fig.add_axes([0.2109, 0.11, 0.604, 0.013])
    plt.rcParams.update({'font.size': 50})
    plt.colorbar(mi,cax = cbar_ax,orientation='horizontal', label = 'Normalized Intensity')
