
import numpy as np

import torch

# +
from torch import nn
from math import sqrt
from radionets.evaluation.utils import  load_pretrained_model
import torch.nn.functional as F

import torch 
import torchvision
import os
import matplotlib.pyplot as plt


# -

class base_maps(nn.Module):
    
    def __init__(self):
        super(base_maps, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv14 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv15 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        self.load_arch()
    def load_arch(self):
        arch = load_pretrained_model('VGG','/net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/build/temp_20.model', 300)
        #arch = load_pretrained_model('VGG', '//net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/build/VGG_test/temp_20.model', 300)
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        pretrained_state_dict = arch.state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        for i, param in enumerate(param_names):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        conv_fc1_weight = pretrained_state_dict['fc1.0.weight'].view(4096, 512, 9, 9)
        conv_fc1_bias = pretrained_state_dict['fc1.0.bias']
        state_dict['conv14.weight'] = decimate(conv_fc1_weight, m = [4, None, 3, 3])
        state_dict['conv14.bias'] = decimate(conv_fc1_bias, m = [4])

        conv_fc2_weight = pretrained_state_dict['fc2.0.weight'].view(4096, 4096, 1, 1)
        conv_fc2_bias = pretrained_state_dict['fc2.0.bias']
        state_dict['conv15.weight'] = decimate(conv_fc2_weight, m = [4, 4, None, None])
        state_dict['conv15.bias'] = decimate(conv_fc2_bias, m = [4])
        
        self.load_state_dict(state_dict)
        print("\n arch loaded \n")
    def forward(self, image):
        out = F.relu(self.conv1(image))  
        out = F.relu(self.conv2(out))  # (N, 64, 300, 300)
        out = self.maxpool1(out)  

        out = F.relu(self.conv3(out))  
        out = F.relu(self.conv4(out))  # (N, 128, 150, 150)
        out = self.maxpool2(out)  

        out = F.relu(self.conv5(out)) 
        out = F.relu(self.conv6(out))  
        out = F.relu(self.conv7(out))  
        fmap7 = out #update            # (N, 256, 75, 75)
        out = self.maxpool3(out) 

        out = F.relu(self.conv8(out))  
        out = F.relu(self.conv9(out))  
        out = F.relu(self.conv10(out))  
        fmap10 = out  # (N, 512, 38, 38)
        out = self.maxpool4(out)  

        out = F.relu(self.conv11(out)) 
        out = F.relu(self.conv12(out))  
        out = F.relu(self.conv13(out))  
        out = self.maxpool5(out)  # (N, 512, 19, 19)
        
        out = F.relu(self.conv14(out))  

        fmap15 = F.relu(self.conv15(out))  # (N, 1024, 19, 19)
        
        base_fmaps = {'fmap7': fmap7,'fmap10': fmap10, 'fmap15':fmap15}
        return base_fmaps


class adv_maps(nn.Module):
    def __init__(self):
        super(adv_maps, self).__init__()

        
        self.conv16 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  
        self.conv17 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  

        self.conv18 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv19 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  

        self.conv20 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        
        self.conv22 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv23 = nn.Conv2d(128, 256, kernel_size=3, padding=0) 
        
        self.init_conv2d()

    def init_conv2d(self):
  
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0)

    def forward(self, fmap15):
      
        out = F.relu(self.conv16(fmap15))  # (N, 256, 19, 19)
        out = F.relu(self.conv17(out))  
        fmap17 = out  # (N, 512, 10, 10)

        out = F.relu(self.conv18(out))  
        out = F.relu(self.conv19(out))  
        fmap19 = out  # (N, 256, 5, 5)

        out = F.relu(self.conv20(out))  
        out = F.relu(self.conv21(out))  
        fmap21 = out  # (N, 256, 3, 3)

        out = F.relu(self.conv22(out))  
        fmap23 = F.relu(self.conv23(out))  
        
        
        fmaps = {'fmap17':fmap17, 'fmap19':fmap19, 'fmap21':fmap21, 'fmap23':fmap23}
        return fmaps


class feature_pyramid(nn.Module):
    def __init__(self):
        super(feature_pyramid, self).__init__()
        
        #self.toplayer = nn.Conv2d()
        #lateral layers
        self.lateral7 = nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0)
        self.lateral10 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0)
        self.lateral15 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, padding = 0)
        self.lateral17 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0)
        self.lateral19 = nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0)
        self.lateral21 = nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0)
        self.lateral23 = nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0)
        #smooth layers to reduce aliasing effects
        self.smooth7 = nn.Conv2d(256,256, kernel_size =3, stride = 1, padding = 1)
        self.smooth10 = nn.Conv2d(256,256, kernel_size =3, stride = 1, padding = 1)
        self.smooth15 = nn.Conv2d(256,256, kernel_size =3, stride = 1, padding = 1)
        self.smooth17 = nn.Conv2d(256,256, kernel_size =3, stride = 1, padding = 1)
        self.smooth19 = nn.Conv2d(256,256, kernel_size =3, stride = 1, padding = 1)
        self.smooth21 = nn.Conv2d(256,256, kernel_size =3, stride = 1, padding = 1)
        
    def upsample_add(self,lateral_map, upper_map):
        _,_,H, W = lateral_map.size()
        return F.upsample(upper_map, size=(H,W), mode = 'bilinear')+lateral_map

    def forward(self,fmap7, fmap10, fmap15, fmap17, fmap19, fmap21, fmap23):
        #maybe add batch norm
        p23 = self.lateral23(fmap23)
        p21 = self.upsample_add(self.lateral21(fmap21),p23)
        p19 = self.upsample_add(self.lateral19(fmap19),p21)
        p17 = self.upsample_add(self.lateral17(fmap17),p19)
        p15 = self.upsample_add(self.lateral15(fmap15),p17)
        p10 = self.upsample_add(self.lateral10(fmap10),p15)
        p7 = self.upsample_add(self.lateral7(fmap7),p10)
        
        p21 = self.smooth21(p21)
        p19 = self.smooth19(p19)
        p17 = self.smooth17(p17)
        p15 = self.smooth15(p15)
        p10 = self.smooth10(p10)
        p7 = self.smooth7(p7)

        return p7, p10, p15, p17, p19, p21, p23



# +
def create_prior_boxes():
    fmap_dims = {    'fmap7' : 75, #update
                     'fmap10': 38, #was 38 with old 37
                     'fmap15': 19, #was 19 with old 18
                     'fmap17': 10, #was 10 with old 9
                     'fmap19': 5,
                     'fmap21': 3,
                     'fmap23': 1}
    maps = list(fmap_dims.keys()) 
   
    scales = {    'fmap7' : 0.02,
                  'fmap10': 0.06,
                  'fmap15': 0.11,
                  'fmap17': 0.16,
                  'fmap19': 0.2,
                  'fmap21': 0.25,
                  'fmap23': 0.3}
    
    
    aspect_ratios = {'fmap7': [1.],
                     'fmap10': [1.,2.,0.5],
                     'fmap15': [1.],
                     'fmap17': [1.],
                     'fmap19': [1.],
                     'fmap21': [1.],
                     'fmap23': [1.]}
    priors = []
    for a, s in enumerate(maps):
        for d in range(fmap_dims[s]):
            for f in range(fmap_dims[s]):
                x = (d + 0.5) / fmap_dims[s]
                y = (f + 0.5) / fmap_dims[s]
                for ratio in aspect_ratios[s]:
                        priors.append([y, x, scales[s] * sqrt(ratio), scales[s] / sqrt(ratio)])
                        
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(scales[s] * scales[maps[a+1]])
                           
                            except IndexError:
                                additional_scale = 1.
                            priors.append([x, y, additional_scale, additional_scale])
    priors = torch.FloatTensor(priors).to('cuda')
    center_to_boundary(priors)
    priors.clamp_(0, 1)
    boundary_to_center(priors)
    return priors

class predconvs(nn.Module):
    def __init__(self, nclasses):
        
        super(predconvs, self).__init__()
        
        self.nclasses = nclasses
        
        #n_boxes = {'fmap10': 4,
        #             'fmap15': 6,
        #             'fmap17': 6,
        #             'fmap19': 6,
        #             'fmap21': 4,
        #             'fmap23': 4}
        
        n_boxes = {  'fmap7': 2,
                     'fmap10': 4,
                     'fmap15': 2,
                     'fmap17': 2,
                     'fmap19': 2,
                     'fmap21': 2,
                     'fmap23': 2}
        self.loc_fmap7 = nn.Conv2d(256, n_boxes['fmap7'] * 4, kernel_size=3, padding=1)
        self.loc_fmap10 = nn.Conv2d(256, n_boxes['fmap10'] * 4, kernel_size=3, padding=1)
        self.loc_fmap15 = nn.Conv2d(256, n_boxes['fmap15'] * 4, kernel_size=3, padding=1)
        self.loc_fmap17 = nn.Conv2d(256, n_boxes['fmap17'] * 4, kernel_size=3, padding=1)
        self.loc_fmap19 = nn.Conv2d(256, n_boxes['fmap19'] * 4, kernel_size=3, padding=1)
        self.loc_fmap21 = nn.Conv2d(256, n_boxes['fmap21'] * 4, kernel_size=3, padding=1)
        self.loc_fmap23 = nn.Conv2d(256, n_boxes['fmap23'] * 4, kernel_size=3, padding=1)
        
        self.cl_fmap7 = nn.Conv2d(256, n_boxes['fmap7'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap10 = nn.Conv2d(256, n_boxes['fmap10'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap15 = nn.Conv2d(256, n_boxes['fmap15'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap17 = nn.Conv2d(256, n_boxes['fmap17'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap19 = nn.Conv2d(256, n_boxes['fmap19'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap21 = nn.Conv2d(256, n_boxes['fmap21'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap23 = nn.Conv2d(256, n_boxes['fmap23'] * nclasses, kernel_size=3, padding=1)
        
        self.init_conv2d()
        
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self,fmap7, fmap10, fmap15, fmap17, fmap19, fmap21, fmap23):
    #stuff
        batch_size = fmap10.size(0)
        l_fmap10 = self.loc_fmap10(fmap10)
        l_fmap10 = l_fmap10.permute(0,2,3,1).contiguous()
        l_fmap10 = l_fmap10.view(batch_size,-1,4)

        l_fmap15 = self.loc_fmap15(fmap15)
        l_fmap15 = l_fmap15.permute(0,2,3,1).contiguous()
        l_fmap15 = l_fmap15.view(batch_size,-1,4)

        l_fmap17 = self.loc_fmap17(fmap17)
        l_fmap17 = l_fmap17.permute(0,2,3,1).contiguous()
        l_fmap17 = l_fmap17.view(batch_size,-1,4)

        l_fmap19 = self.loc_fmap19(fmap19)
        l_fmap19 = l_fmap19.permute(0,2,3,1).contiguous()
        l_fmap19 = l_fmap19.view(batch_size,-1,4)

        l_fmap21 = self.loc_fmap21(fmap21)
        l_fmap21 = l_fmap21.permute(0,2,3,1).contiguous()
        l_fmap21 = l_fmap21.view(batch_size,-1,4)

        l_fmap23 = self.loc_fmap23(fmap23)
        l_fmap23 = l_fmap23.permute(0,2,3,1).contiguous()
        l_fmap23 = l_fmap23.view(batch_size,-1,4)
        
        l_fmap7 = self.loc_fmap7(fmap7)
        l_fmap7 = l_fmap7.permute(0,2,3,1).contiguous()
        l_fmap7 = l_fmap7.view(batch_size,-1,4)
        
        c_fmap10 = self.cl_fmap10(fmap10)
        c_fmap10 = c_fmap10.permute(0,2,3,1).contiguous()
        c_fmap10 = c_fmap10.view(batch_size,-1,self.nclasses)

        c_fmap15 = self.cl_fmap15(fmap15)
        c_fmap15 = c_fmap15.permute(0,2,3,1).contiguous()
        c_fmap15 = c_fmap15.view(batch_size,-1,self.nclasses)

        c_fmap17 = self.cl_fmap17(fmap17)
        c_fmap17 = c_fmap17.permute(0,2,3,1).contiguous()
        c_fmap17 = c_fmap17.view(batch_size,-1,self.nclasses)

        c_fmap19 = self.cl_fmap19(fmap19)
        c_fmap19 = c_fmap19.permute(0,2,3,1).contiguous()
        c_fmap19 = c_fmap19.view(batch_size,-1,self.nclasses)

        c_fmap21 = self.cl_fmap21(fmap21)
        c_fmap21 = c_fmap21.permute(0,2,3,1).contiguous()
        c_fmap21 = c_fmap21.view(batch_size,-1,self.nclasses)

        c_fmap23 = self.cl_fmap23(fmap23)
        c_fmap23 = c_fmap23.permute(0,2,3,1).contiguous()
        c_fmap23 = c_fmap23.view(batch_size,-1,self.nclasses)

        c_fmap7 = self.cl_fmap7(fmap7)
        c_fmap7 = c_fmap7.permute(0,2,3,1).contiguous()
        c_fmap7 = c_fmap7.view(batch_size,-1,self.nclasses)
        
        locs = torch.cat([l_fmap7, l_fmap10, l_fmap15, l_fmap17, l_fmap19, l_fmap21, l_fmap23], dim = 1)
        classes_scores = torch.cat([c_fmap7, c_fmap10, c_fmap15, c_fmap17, c_fmap19, c_fmap21, c_fmap23], dim = 1)    
        return locs, classes_scores
    
class SSD300(nn.Module):
    
    def __init__(self, nclasses):
        
        super(SSD300, self).__init__()
        
        self.nclasses = nclasses
        
        self.base = base_maps()
        self.adv = adv_maps()
        self.pred_convs = predconvs(nclasses)
        
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1)) 
        self.rescale_factors7 = nn.Parameter(torch.FloatTensor(1, 256, 1, 1))  
        nn.init.constant_(self.rescale_factors, 20)
        nn.init.constant_(self.rescale_factors7, 20)
        self.priors_cxcy = create_prior_boxes()
        self.feature_pyramid = feature_pyramid()

    def forward(self, image):
        
        

        bmaps = self.base(image)
        fmap10 = bmaps['fmap10'] #[256, 38, 38]
        fmap15 = bmaps['fmap15']
        fmap7 = bmaps['fmap7']
        #fmap_13 = bmaps['fmap_13']
        norm7 = fmap7.pow(2).sum(dim=1, keepdim=True).sqrt()
        norm = fmap10.pow(2).sum(dim=1, keepdim=True).sqrt()
        fmap10 = fmap10 / norm
        fmap10 = fmap10 * self.rescale_factors
        
        fmap7 = fmap7 / norm7
        fmap7 = fmap7 * self.rescale_factors7
        
        amaps = self.adv(fmap15)
        fmap17 = amaps['fmap17']
        fmap19 = amaps['fmap19']
        fmap21 = amaps['fmap21']
        fmap23 = amaps['fmap23']
        fmap7, fmap10, fmap15, fmap17, fmap19, fmap21, fmap23 = self.feature_pyramid(fmap7, fmap10, fmap15, fmap17, fmap19, fmap21, fmap23)
        locs, classes_scores = self.pred_convs(fmap7, fmap10, fmap15, fmap17, fmap19, fmap21, fmap23)
        
        return locs, classes_scores
    
    def object_detection(self, locs, class_scores, priors, min_score=0.01, max_overlap=0.45,top_k=200):
        
        batch_size = locs.size(0)
        n_priors = priors.size(0)
        
        classes_scores = F.softmax(class_scores, dim = 2)
        all_predicted_boxes = list()
        all_predicted_labels = list()
        all_predicted_scores = list()
        assert n_priors == locs.size(1) == classes_scores.size(1)
        for i in range(batch_size):
            boundary_locs = center_to_boundary(offset_to_center(locs[i], priors))
            predicted_boxes = list()
            predicted_labels = list()
            predicted_scores = list()
            max_scores, best_label = classes_scores[i].max(dim=1)
            for c in range(0, self.nclasses-1):
                c_scores = classes_scores[i][:,c]
                score_above_min = c_scores > min_score
                n_above_min = score_above_min.sum().item()
                if n_above_min == 0:
                    continue
                c_scores = c_scores[score_above_min]
                c_boundary_locs = boundary_locs[score_above_min]
                
                c_scores, sort_ind = c_scores.sort(dim = 0, descending = True)
                c_boundary_locs = c_boundary_locs[sort_ind]
                
                overlap = jaccard(c_boundary_locs, c_boundary_locs)
                suppress = torch.zeros((n_above_min)).bool().to('cuda') 
                for box in range(c_boundary_locs.size(0)):
                    if suppress[box] == 1:
                        continue

                    suppress = suppress | (overlap[box] > max_overlap)

                    suppress[box] = 0
               
                predicted_boxes.append(c_boundary_locs[~suppress])
                predicted_labels.append(torch.LongTensor((~suppress).sum().item()*[c]).to('cuda'))
                predicted_scores.append(c_scores[~suppress])
            if len(predicted_boxes) == 0:
                predicted_boxes.append(torch.FloatTensor([[0.,0.,1.,1.]]).to('cuda'))
                predicted_labels.append(torch.LongTensor([4]).to('cuda')) #nodiff
                predicted_scores.append(torch.FloatTensor([0.]).to('cuda'))
            
            predicted_boxes = torch.cat(predicted_boxes, dim = 0)
            predicted_labels = torch.cat(predicted_labels, dim = 0)
            predicted_scores = torch.cat(predicted_scores, dim = 0)
            num_objects = predicted_scores.size(0)
            
            if num_objects > top_k: 
                predicted_scores, sort_ind = predicted_scores.sort(dim = 0, descending = True)
                predicted_scores = predicted_scores[:top_k]
                predicted_boxes = predicted_boxes[:top_k]
                predicted_labels = predicted_labels[:top_k]
           
            all_predicted_boxes.append(predicted_boxes)
            all_predicted_labels.append(predicted_labels)
            all_predicted_scores.append(predicted_scores)
        
        return all_predicted_boxes, all_predicted_labels, all_predicted_scores
    
def center_to_boundary(coord):
    return torch.cat([coord[:,:2]-(coord[:,2:]/2),
                      coord[:,:2]+(coord[:,2:]/2)], 1)
def boundary_to_center(coord):
    return torch.cat([(coord[:,2:]+coord[:,:2])/2,
                     coord[:,2:]-coord[:,:2]], 1)

    #"The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155"
def offset_to_center(coded_box_coord, prior_coord):
    return torch.cat([coded_box_coord[:,:2] * prior_coord[:,2:]/10 + prior_coord[:,:2],
                      torch.exp(coded_box_coord[:,2:]/5)*prior_coord[:,2:]],1) 

def center_to_offset(box_coord, prior_coord):
    return torch.cat([(box_coord[:, :2] - prior_coord[:,:2])/(prior_coord[:,2:]/10),
                     torch.log(box_coord[:,2:] / prior_coord[:,2:])* 5], 1)
    
def jaccard(boxes1, boxes2):
    
    low_bound = torch.max(boxes1[:,:2].unsqueeze(1), boxes2[:,:2].unsqueeze(0))
    up_bound = torch.min(boxes1[:,2:].unsqueeze(1), boxes2[:,2:].unsqueeze(0))
    
    intersect_dims = torch.clamp(up_bound - low_bound, min = 0)
    intersect =  intersect_dims[:,:,0]*intersect_dims[:, :, 1]

    area_boxes1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area_boxes2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    
    union = area_boxes1.unsqueeze(1) +area_boxes2.unsqueeze(0) - intersect
    
    return intersect/union
def decimate(tensor, m):
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

class old_predconvs(nn.Module):
    def __init__(self, nclasses):
        #some changes
        super(predconvs, self).__init__()
        
        self.nclasses = nclasses
        
        n_boxes = {'fmap10': 4,
                     'fmap_10': 6,
                     'fmap_13': 6,
                     'fmap_15': 6,
                     'fmap_17': 4,
                     'fmap_19': 4}
        
        self.loc_fmap10 = nn.Conv2d(512, n_boxes['fmap_7'] * 4, kernel_size=3, padding=1)
        self.loc_fmap_10 = nn.Conv2d(1024, n_boxes['fmap_10'] * 4, kernel_size=3, padding=1)
        self.loc_fmap_13 = nn.Conv2d(512, n_boxes['fmap_13'] * 4, kernel_size=3, padding=1)#channels were different
        self.loc_fmap_15 = nn.Conv2d(256, n_boxes['fmap_15'] * 4, kernel_size=3, padding=1)#same here
        self.loc_fmap_17 = nn.Conv2d(256, n_boxes['fmap_17'] * 4, kernel_size=3, padding=1)
        self.loc_fmap_19 = nn.Conv2d(256, n_boxes['fmap_19'] * 4, kernel_size=3, padding=1)
        
        self.cl_fmap_7 = nn.Conv2d(512, n_boxes['fmap_7'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap_10 = nn.Conv2d(1024, n_boxes['fmap_10'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap_13 = nn.Conv2d(512, n_boxes['fmap_13'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap_15 = nn.Conv2d(256, n_boxes['fmap_15'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap_17 = nn.Conv2d(256, n_boxes['fmap_17'] * nclasses, kernel_size=3, padding=1)
        self.cl_fmap_19 = nn.Conv2d(256, n_boxes['fmap_19'] * nclasses, kernel_size=3, padding=1)
        
        self.init_conv2d()
        
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, fmap7, fmap10, fmap13, fmap15, fmap17, fmap19):
    
        batch_size = fmap7.size(0)
        l_fmap7 = self.loc_fmap_7(fmap7)
        l_fmap7 = l_fmap7.permute(0,2,3,1).contiguous()
        l_fmap7 = l_fmap7.view(batch_size,-1,4)

        l_fmap10 = self.loc_fmap_10(fmap10)
        l_fmap10 = l_fmap10.permute(0,2,3,1).contiguous()
        l_fmap10 = l_fmap10.view(batch_size,-1,4)

        l_fmap13 = self.loc_fmap_13(fmap13)
        l_fmap13 = l_fmap13.permute(0,2,3,1).contiguous()
        l_fmap13 = l_fmap13.view(batch_size,-1,4)

        l_fmap15 = self.loc_fmap_15(fmap15)
        l_fmap15 = l_fmap15.permute(0,2,3,1).contiguous()
        l_fmap15 = l_fmap15.view(batch_size,-1,4)

        l_fmap17 = self.loc_fmap_17(fmap17)
        l_fmap17 = l_fmap17.permute(0,2,3,1).contiguous()
        l_fmap17 = l_fmap17.view(batch_size,-1,4)

        l_fmap19 = self.loc_fmap_19(fmap19)
        l_fmap19 = l_fmap19.permute(0,2,3,1).contiguous()
        l_fmap19 = l_fmap19.view(batch_size,-1,4)

        c_fmap7 = self.cl_fmap_7(fmap7)
        c_fmap7 = c_fmap7.permute(0,2,3,1).contiguous()
        c_fmap7 = c_fmap7.view(batch_size,-1,self.nclasses)

        c_fmap10 = self.cl_fmap_10(fmap10)
        c_fmap10 = c_fmap10.permute(0,2,3,1).contiguous()
        c_fmap10 = c_fmap10.view(batch_size,-1,self.nclasses)

        c_fmap13 = self.cl_fmap_13(fmap13)
        c_fmap13 = c_fmap13.permute(0,2,3,1).contiguous()
        c_fmap13 = c_fmap13.view(batch_size,-1,self.nclasses)

        c_fmap15 = self.cl_fmap_15(fmap15)
        c_fmap15 = c_fmap15.permute(0,2,3,1).contiguous()
        c_fmap15 = c_fmap15.view(batch_size,-1,self.nclasses)

        c_fmap17 = self.cl_fmap_17(fmap17)
        c_fmap17 = c_fmap17.permute(0,2,3,1).contiguous()
        c_fmap17 = c_fmap17.view(batch_size,-1,self.nclasses)

        c_fmap19 = self.cl_fmap_19(fmap19)
        c_fmap19 = c_fmap19.permute(0,2,3,1).contiguous()
        c_fmap19 = c_fmap19.view(batch_size,-1,self.nclasses)

        locs = torch.cat([l_fmap7, l_fmap10, l_fmap13, l_fmap15, l_fmap17, l_fmap19], dim = 1)
        classes_scores = torch.cat([c_fmap7, c_fmap10, c_fmap13, c_fmap15, c_fmap17, c_fmap19], dim = 1)    
        return locs, classes_scores
