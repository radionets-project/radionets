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
    return (torch.tensor(all_boxes)/(img_size*10)), all_labels, torch.tensor(all_scores)

def mosaic_clean(mos_boxes,labels,scores):
    bboxes = torch.tensor(box_coord_inv(mos_boxes))
    num_classes = 4
    max_overlap = 0.1
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
            if suppress[box] == 1:
                continue

            suppress = suppress | (overlap[box] > max_overlap)

            suppress[box] = 0
        predicted_boxes.append(c_boxes[~suppress])
        predicted_labels.append(c_labels[~suppress])
    
    #final_boxes = np.concatenate((predicted_boxes[0]))

    #final_labels = np.concatenate((predicted_labels[0]))    
    final_boxes = predicted_boxes[0]#np.concatenate((predicted_boxes[0],predicted_boxes[1],predicted_boxes[2],predicted_boxes[3]))
    final_labels = predicted_labels[0]#np.concatenate((predicted_labels[0],predicted_labels[1],predicted_labels[2],predicted_labels[3]))
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
            predrect = patches.Rectangle((all_boxes[k][0], all_boxes[k][1]), all_boxes[k][2], all_boxes[k][3], linewidth=1, edgecolor=color,
                                         facecolor='none')
            #ax.text(all_boxes[k][0],(all_boxes[k][1]+all_boxes[k][2]-20),predicted_label, color = 'k',fontsize=8,backgroundcolor = color)
            ax.add_patch(predrect)

    ax.imshow(img,cmap = 'gist_heat')
    return (torch.tensor(all_boxes)/(img_size*10)), all_labels, torch.tensor(all_scores)
