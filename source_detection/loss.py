# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from torch import nn
import torch
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
from source_detection.model import center_to_boundary, center_to_offset, boundary_to_center, jaccard,offset_to_center
import torchvision

class detectionLoss(nn.Module):                 #0.6                 #9         #20.
    def __init__(self, priors_cxcy, threshold = 0.5, neg_pos_ratio = 3, alpha = 1.):
        
        super(detectionLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = center_to_boundary(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce = False)
    
    def forward(self, predicted_locs, predicted_scores, data_locs, data_labels):
        
        batch_size = predicted_locs.size(0)
        n_classes = predicted_scores.size(2)
        n_priors = self.priors_cxcy.size(0)
        assert n_priors ==  predicted_locs.size(1) == predicted_scores.size(1)
        
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype = torch.float).to('cuda')
        true_classes = torch.zeros((batch_size, n_priors), dtype = torch.long).to('cuda')
        
        for image_i in range(batch_size):
            n_objects = data_locs[image_i][0].size(0)
            overlap = jaccard(data_locs[image_i][0], self.priors_xy) #overlap of the boxes in this image with the priors
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            #overlap has shape [a,s,f,g,....]
                              #[h,d,g,h,....].... each entry is the overlap of one true box with all the priors.
                                #each row describes one object. Max gives the maximum overlap value and the index of the object.
            
            _, prior_for_each_object = overlap.max(dim = 1)
            
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to('cuda')
            
            overlap_for_each_prior[prior_for_each_object] = 1.
            label_for_each_prior = data_labels[image_i][0][0][object_for_each_prior]#very ugly shapes watch out
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 4 #nodiff
            true_classes[image_i] = label_for_each_prior
            
            true_locs[image_i] = center_to_offset(boundary_to_center(data_locs[image_i][0][object_for_each_prior]), self.priors_cxcy)
        positive_priors = true_classes != 4 #nodiff
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])
        n_positives = positive_priors.sum(dim = 1)
        n_hard_negatives = self.neg_pos_ratio * n_positives
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)
        
        conf_loss_pos = conf_loss_all[positive_priors]
        
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0
        conf_loss_neg, _ = conf_loss_neg.sort(dim = 1, descending = True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to('cuda')
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        
        conf_loss = (conf_loss_hard_neg.sum()+conf_loss_pos.sum())/n_positives.sum().float()
        return conf_loss + self.alpha * loc_loss
# -


