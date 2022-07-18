from  FPNeval import detect_sources, box_coord,image_detection,box_coord_inv, annotate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from matplotlib.pyplot import figure
from FPN import jaccard
import torch.nn.functional as F

label_map = ('pointlike gaussian', 'diffuse gaussian', 'diamond', 'square', 'background')
color_map = ('w', 'r', 'pink', 'r','brown')


def mosaic_detection(model_path,img, nclasses, img_size = 300): #classes without background
    all_boxes = np.array([0,0,0,0])
    all_labels = np.array([])
    all_scores = np.array([])
    for tile_y in range(10):
        for tile_x in range(10):
            predboxes,predlabels,preds = image_detection(model_path,img[img_size*tile_y:img_size*tile_y+img_size,img_size*tile_x:img_size*tile_x+img_size])
            if predlabels[0].max().item() != 4: #and predlabels[0].max().item()!=4 :
                for n in range(predboxes[0].shape[0]):
                    predx, predy, predw, predh = box_coord(predboxes[0][n],img_size)
                    boxes = np.array([predx+tile_x*img_size, predy+tile_y*img_size, predw, predh])
                    all_boxes = np.vstack((all_boxes,boxes))           
                all_labels = np.concatenate((all_labels,predlabels[0].cpu().detach().numpy()), axis = 0)
                all_scores = np.concatenate((all_labels,preds[0].cpu().detach().numpy()), axis = 0)
                testbox = predboxes[0][n]
    all_boxes = np.delete(all_boxes, (0), axis=0)
    

    fig, ax = plt.subplots(1,1, figsize=(50,40))
    for k in range(len(all_boxes)):
            predicted_label = label_map[int(all_labels[k])]
            color = color_map[int(all_labels[k])]
            predrect = patches.Rectangle((all_boxes[k][0], all_boxes[k][1]), all_boxes[k][2], all_boxes[k][3], linewidth=1, edgecolor=color,
                                         facecolor='none')
            #ax.text(all_boxes[k][0],(all_boxes[k][1]+all_boxes[k][2]-20),predicted_label, color = 'k',fontsize=8,backgroundcolor = color)
            ax.add_patch(predrect)

    ax.imshow(img,cmap = 'gist_heat')
    ax.axis('off')
    return (torch.tensor(all_boxes)/(img_size*10)), all_labels, torch.tensor(all_scores)

def mosaic_clean(mos_boxes,labels,scores):
    bboxes = torch.tensor(box_coord_inv(mos_boxes))
    num_classes = 4
    max_overlap = 0
    predicted_boxes = list()
    predicted_labels = list()
    for c in range(num_classes):
        label_mask = np.where(labels == c)
        c_boxes = bboxes[label_mask]
        c_labels = labels[label_mask]
        if c_labels.shape[0] == 0:
            continue
        c_scores = scores[c_labels]
        c_scores, sort_ind = c_scores.sort(dim = 0, descending = True)
        c_boxes= c_boxes[sort_ind]
        overlap = jaccard(c_boxes,c_boxes) 
        suppress = torch.zeros((c_boxes.size(0))).bool()
        for box in range(c_boxes.size(0)):
            #if suppress[box] == 1:
             #   continue

            suppress = suppress | (overlap[box] > max_overlap)

            suppress[box] = 0
        predicted_boxes.append(c_boxes[~suppress])
        predicted_labels.append(c_labels[~suppress])
    
    #final_boxes = np.concatenate((predicted_boxes[0]))

    #final_labels = np.concatenate((predicted_labels[0]))  
    if len(predicted_boxes) == 4:
        final_boxes = np.concatenate((predicted_boxes[0],predicted_boxes[1],predicted_boxes[2],predicted_boxes[3]))
        final_labels = np.concatenate((predicted_labels[0],predicted_labels[1],predicted_labels[2],predicted_labels[3]))
    else:
        final_boxes = np.concatenate((predicted_boxes[0],predicted_boxes[1],predicted_boxes[2]))
        final_labels = np.concatenate((predicted_labels[0],predicted_labels[1],predicted_labels[2]))
    return final_boxes, final_labels


def upscale_mosaic_detection(model_path,img, nclasses, img_size = 150): #classes without background
    all_boxes = np.array([0,0,0,0])
    all_labels = np.array([])
    all_scores = np.array([])
    for tile_y in range(img.shape[0]//img_size):
        print('row:', tile_y)
        for tile_x in range(img.shape[0]//img_size):
            dimg = img[img_size*tile_y:img_size*tile_y+img_size,img_size*tile_x:img_size*tile_x+img_size]
            #print(1, img_size*tile_y,img_size*tile_y+img_size,img_size*tile_x,img_size*tile_x+img_size)
           
            dimg = torch.FloatTensor(dimg)
            #print('test',img_size*tile_y,img_size*tile_y+img_size)
            dimg = dimg.unsqueeze(0)
            #print(2, tile_x)
            dimg = dimg.unsqueeze(0)
            #print(3, dimg.shape)
            uimg = F.interpolate(dimg, size = (300,300) , mode = 'bilinear')
            #print(uimg.shape)
            uimg = uimg[0][0]
            uimg = uimg/uimg.max()
            predboxes,predlabels,preds = image_detection(model_path,uimg)
            if predlabels[0].max().item() != nclasses:
                for n in range(predboxes[0].shape[0]):
                    predx, predy, predw, predh = box_coord(predboxes[0][n],img_size)
                    boxes = np.array([predx+tile_x*img_size, predy+tile_y*img_size, predw, predh])
                    all_boxes = np.vstack((all_boxes,boxes))           
                all_labels = np.concatenate((all_labels,predlabels[0].cpu().detach().numpy()), axis = 0)
                all_scores = np.concatenate((all_labels,preds[0].cpu().detach().numpy()), axis = 0)
                testbox = predboxes[0][n]
            del dimg, uimg
    all_boxes = np.delete(all_boxes, (0), axis=0)
    

    fig, ax = plt.subplots(1,1, figsize=(50,40))
    for k in range(len(all_boxes)):
            predicted_label = label_map[int(all_labels[k])]
            color = color_map[int(all_labels[k])]
            predrect = patches.Rectangle((all_boxes[k][0], all_boxes[k][1]), all_boxes[k][2], all_boxes[k][3], linewidth=1, edgecolor=color,
                                         facecolor='none')
            #ax.text(all_boxes[k][0],(all_boxes[k][1]+all_boxes[k][2]-20),predicted_label, color = 'k',fontsize=8,backgroundcolor = color)
            ax.add_patch(predrect)

    ax.imshow(img,cmap = 'gist_heat')
    return (torch.tensor(all_boxes)/(img_size*10)), all_labels, torch.tensor(all_scores)


def shifted_mosaic_detection(model_path,img, nclasses, img_size = 300): #classes without background
    all_boxes = np.array([0,0,0,0])
    all_labels = np.array([])
    all_scores = np.array([])
    for tile_y in range(10):
        for tile_x in range(10):
            predboxes,predlabels,preds = image_detection(model_path,img[img_size*tile_y:img_size*tile_y+img_size,img_size*tile_x:img_size*tile_x+img_size])
            if predlabels[0].max().item() != 4: #and predlabels[0].max().item()!=4 :
                for n in range(predboxes[0].shape[0]):
                    predx, predy, predw, predh = box_coord(predboxes[0][n],img_size)
                    boxes = np.array([predx+tile_x*img_size, predy+tile_y*img_size, predw, predh])
                    all_boxes = np.vstack((all_boxes,boxes))           
                all_labels = np.concatenate((all_labels,predlabels[0].cpu().detach().numpy()), axis = 0)
                all_scores = np.concatenate((all_labels,preds[0].cpu().detach().numpy()), axis = 0)
                testbox = predboxes[0][n]
    for tile_y in range(9):
        for tile_x in range(9):
            predboxes,predlabels,preds = image_detection(model_path,img[img_size//2+img_size*tile_y:img_size*tile_y+img_size+img_size//2,img_size//2+img_size*tile_x:img_size*tile_x+img_size+img_size//2])
            if predlabels[0].max().item() != 4: #and predlabels[0].max().item()!=4 :
                for n in range(predboxes[0].shape[0]):
                    predx, predy, predw, predh = box_coord(predboxes[0][n],img_size)
                    boxes = np.array([predx+tile_x*img_size+img_size//2, predy+tile_y*img_size+img_size//2, predw, predh])
                    all_boxes = np.vstack((all_boxes,boxes))           
                all_labels = np.concatenate((all_labels,predlabels[0].cpu().detach().numpy()), axis = 0)
                all_scores = np.concatenate((all_labels,preds[0].cpu().detach().numpy()), axis = 0)
                testbox = predboxes[0][n]
    all_boxes = np.delete(all_boxes, (0), axis=0)
    

    fig, ax = plt.subplots(1,1, figsize=(50,40))
    for k in range(len(all_boxes)):
            predicted_label = label_map[int(all_labels[k])]
            color = color_map[int(all_labels[k])]
            predrect = patches.Rectangle((all_boxes[k][0], all_boxes[k][1]), all_boxes[k][2], all_boxes[k][3], linewidth=3, edgecolor=color,
                                         facecolor='none')
            #ax.text(all_boxes[k][0],(all_boxes[k][1]+all_boxes[k][2]-20),predicted_label, color = 'k',fontsize=8,backgroundcolor = color)
            ax.add_patch(predrect)

    ax.imshow(img,cmap = 'gist_heat')
    return (torch.tensor(all_boxes)/(img_size*10)), all_labels, torch.tensor(all_scores)


def mosaicTPFP(pred_boxes, pred_labels, true_boxes, true_labels):
        pred_boxes = torch.tensor(pred_boxes).unsqueeze(0)
        pred_labels = torch.tensor(pred_labels).unsqueeze(0)
        true_boxes = torch.tensor(true_boxes)
        true_labels = torch.tensor(true_labels)
        TP = 0
        FP = 0
        n_classes = 4
        
        #BEWARE BELOW
        for c in range(0, n_classes):
            mask = (true_labels == c)[0].T[0]
            pred_mask = (pred_labels == c)[0]
            true_class_boxes = true_boxes[0][mask] 
            
            # Keep track of which true objects with this class have already been 'detected'
            # So far, none
            true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(
                'cuda')  # (n_class_objects)

            # Extract only detections with this class
            pred_class_boxes = pred_boxes[pred_mask]  # (n_class_detections, 4)
                
            n_class_detections = pred_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to('cuda')  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to('cuda')  # (n_class_detections)
            for d in range(n_class_detections):
                this_detection_box = pred_class_boxes[d].unsqueeze(0)  # (1, 4)
                object_boxes = true_class_boxes  # (n_class_objects_in_img)
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                overlaps = jaccard(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[ind]
                if max_overlap.item() >= 0.3:
                     if true_class_boxes_detected[ original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        # Otherwise, it's a false positive (since this object is already accounted for)
                     else:
                        false_positives[d] = 1
                else:
                    false_positives[d] = 1
            cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
            cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
            TP += cumul_true_positives.max().item()
            FP += cumul_false_positives.max().item()
        return TP, FP
