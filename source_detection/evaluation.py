# +
import train
import torch
from radionets.evaluation.utils import  load_pretrained_model, eval_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import model
from source_detection.model import center_to_boundary
from radionets.dl_framework.data import get_bundles
from tqdm import tqdm

class_labels = ('pointlike gaussian', 'diffuse gaussian', 'diamond', 'square', 'background')
color_map = ('y', 'g', 'w', 'r','brown')
label_map = {k: v for v, k in enumerate(class_labels)}
rev_label_map = {v: k for k, v in label_map.items()} 
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
                                     min_score = 0.5, max_overlap = 0.45, top_k = 10)
    fig, (ax2) = plt.subplots(1,1,figsize=(12,8))
    for j in range(len(eval_dataset[n][1][0])):
        true_label = class_labels[eval_dataset[n][2][0][j].item()]
        color = color_map[eval_dataset[n][2][0][j].item()]
        trux, truy, truw, truh = box_coord(eval_dataset[n][1][0][j],img_size)
        trurect = patches.Rectangle((trux, truy), truw, truh, linewidth=1, edgecolor=color, facecolor='none')
        #ax1.text(trux,(truy+truh-7),true_label, color = 'k',fontsize=8,backgroundcolor = color)
        #ax1.add_patch(trurect)
    
    for k in range(len(predl[n])):
        predicted_label = class_labels[predl[n][k].item()]
        color = color_map[predl[n][k].item()]
        predx, predy, predw, predh = box_coord(predb[n][k],img_size)
        predrect = patches.Rectangle((predx, predy), predw, predh, linewidth=1, edgecolor=color,
                                     facecolor='none')
        #ax2.text(predx,(predy+predh-7),predicted_label, color = 'k',fontsize=8,backgroundcolor = color)
        ax2.add_patch(predrect)
    
    #ax1.imshow(eval_dataset[n][0].squeeze(0))
    img = ax2.imshow(eval_dataset[n][0].squeeze(0))
    fig.colorbar(img)
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

def mAPeval(checkpoint_path, data_path):
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
                                     min_score = 0.5, max_overlap = 0.45, top_k = 10)
            boxes = [boxes[b][0] for b in range(len(boxes))]
            labels = [labels[l][0][0] for l in range(len(labels))]
            pred_boxes.extend(predb)
            pred_labels.extend(predl)
            pred_scores.extend(preds)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
        
        APs, mAP = calculate_mAP(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels)
    print(APs)
    print('\nMean Average Precision: %.3f' %mAP)

def calculate_mAP(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels):
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
        
        #BEWARE BELOW
        for c in range(0, n_classes-1):
            print(c)
            true_class_images = true_images[true_labels == c] 
            true_class_boxes = true_boxes[true_labels == c] 
            
            # Keep track of which true objects with this class have already been 'detected'
            # So far, none
            true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(
                'cuda')  # (n_class_objects)

            # Extract only detections with this class
            pred_class_images = pred_images[pred_labels == c]  # (n_class_detections)
            pred_class_boxes = pred_boxes[pred_labels == c]  # (n_class_detections, 4)
            pred_class_scores = pred_scores[pred_labels == c]  # (n_class_detections)
            n_class_detections = pred_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            # Sort detections in decreasing order of confidence/scores
            pred_class_scores, sort_ind = torch.sort(pred_class_scores, dim=0, descending=True)  # (n_class_detections)
            pred_class_images = pred_class_images[sort_ind]  # (n_class_detections)
            pred_class_boxes = pred_class_boxes[sort_ind]  # (n_class_detections, 4)

            # In the order of decreasing scores, check if true or false positive
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to('cuda')  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to('cuda')  # (n_class_detections)
            for d in range(n_class_detections):
                this_detection_box = pred_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = pred_class_images[d]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                # Find maximum overlap of this detection with objects in this image of this class
                overlaps = model.jaccard(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

                # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                # We need 'original_ind' to update 'true_class_boxes_detected'

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if max_overlap.item() > 0.5:
                     if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        # Otherwise, it's a false positive (since this object is already accounted for)
                     else:
                        false_positives[d] = 1
                # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                else:
                    false_positives[d] = 1

            # Compute cumulative precision and recall at each detection in the order of decreasing scores
            cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
            cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
            cumul_precision = cumul_true_positives / (
                    cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
            cumul_recall = cumul_true_positives / len(true_class_images)  # (n_class_detections)

            # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
            recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to('cuda')  # (11)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if recalls_above_t.any():
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.
            average_precisions[c] = precisions.mean()  # c is in [1, n_classes - 1]

        # Calculate Mean Average Precision (mAP)
        mean_average_precision = average_precisions.mean().item()

        # Keep class-wise average precisions in a dictionary
        average_precisions = {rev_label_map[c]: v for c, v in enumerate(average_precisions.tolist())}

        return average_precisions, mean_average_precision
