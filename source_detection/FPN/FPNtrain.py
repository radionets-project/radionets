#just a copy of old train
import torch
import h5py
import numpy as np
from radionets.dl_framework.data import get_bundles
from FPNloss import detectionLoss, FocalLoss
from FPN import SSD300
from tqdm import tqdm
import matplotlib.pyplot as plt

path = get_bundles('/net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/july/mixed')
iterations = 120000
n_classes = 3 #nodiff
checkpoint = None
#checkpoint = '/net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/june/mixedcheckpoints/checkpoint_ssd300.pth.tar'
batch_size = 32
workers = 4
lr = 1e-9
decay_lr_at = [80000,120000]
decay_lr_to = 0.1
#increase_lr_at = [10,30000]
#increase_lr_to = 10
#increase_lr_at2 = [100,80000]
#increase_lr_to2 = 100
momentum = 0.9
weight_decay = 5e-4
grad_clip = None

class detect_dataset:
    
    def __init__(self, bundle_path):
        
        self.bundles = bundle_path
        self.num_img = len(h5py.File(self.bundles[0]))//3
        
    def __getitem__(self, i):
        
        x = self.open_image('x', i)
        y = self.open_boxes('y', i)
        z = self.open_labels('z', i)
        return x,y,z
    def __len__(self):
        return len(self.bundles)*self.num_img
    def open_image(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])
        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)
        indices, _ = torch.sort(i)
        bundle = indices // self.num_img
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)
        #print('bundle:',bundle)
        bundle_paths = [
                h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
            ]
        bundle_paths_str = list(map(str, bundle_paths))
        data = torch.FloatTensor( ###
            [
                #print(bund[var+str(int(img))].shape)
                bund[var+str(int(img))][0] #VGG
                for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                for img in image[
                    bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                ]
            ]
        )
        return data
    def open_boxes(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])
        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)
        indices, _ = torch.sort(i)
        bundle = indices // self.num_img
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)
        bundle_paths = [
                h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
            ]
        bundle_paths_str = list(map(str, bundle_paths))
        
        data = [
            torch.FloatTensor(bund[var+str(int(img))][:]).to('cuda')
            for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                for img in image[
                    bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                ]
        ]
        return data
    def open_labels(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])
        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)
        indices, _ = torch.sort(i)
        bundle = indices // self.num_img
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)
        bundle_paths = [
                h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
            ]
        bundle_paths_str = list(map(str, bundle_paths))
        
        data = [
            torch.tensor(bund[var+str(int(img))][:]).long().squeeze(-1).to('cuda')
            for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                for img in image[
                    bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                ]
        ]
        return data
    def collate_fn(self, batch):
        
        images = list()
        bboxes = list()
        labels = list()
        
        for b in batch:
            images.append(b[0])
            bboxes.append(b[1])
            labels.append([b[2]])
        images = torch.stack(images, dim=0)
        
        return images, bboxes, labels 
def main():
    
    global start_epoch, epoch, checkpoint, decay_lr_at
    
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(nclasses = n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
       
        
    
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']+1
        print('loaded checkpoint')
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    model = model.to('cuda')
    
    loss_function = FocalLoss(priors_cxcy = model.priors_cxcy).to('cuda')#detectionLoss(priors_cxcy = model.priors_cxcy).to('cuda')
    
    
               
    train_dataset = detect_dataset(path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size,
                                               shuffle = True,
                                               collate_fn = train_dataset.collate_fn)
    
    epochs = iterations//(len(train_dataset)//batch_size) 
    decay_lr_at = [it // (len(train_dataset)//batch_size) for it in decay_lr_at]
    
    for epoch in tqdm(range(start_epoch, epochs)): 
        
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to) 
        #if epoch in increase_lr_at:
           # adjust_learning_rate(optimizer, increase_lr_to) 
        #if epoch in increase_lr_at2:
            #adjust_learning_rate(optimizer, increase_lr_to2) 
        train(train_loader, model, loss_function, optimizer, epoch)
        
        print("Epoch:", epoch)
        
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer,'/net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/july/focal/checkpoint_ssd300' + '_e' + str(epoch)+'.pth.tar')
        save_checkpoint(epoch, model, optimizer,'/net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/july/focal/checkpoint_ssd300.pth.tar')# apparently not defined


def train(data_loader, model, loss_function, optimizer, epochs):
    
    model.train()
    losses = np.zeros(940)
    for i, (images, boxes, labels) in enumerate(data_loader):
        images = images.to('cuda')
        
        predicted_locs, predicted_classes_scores= model(images)
        loss = loss_function(predicted_locs, predicted_classes_scores,
                    boxes, labels)
        
        
            
        losses[i] = loss
        #print('i', i, 'Loss:',loss)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        
        optimizer.step()
        
    print('Average Loss', np.average(losses))
    f = open('/net/big-tank/POOL/users/pblomenkamp/radionets/objectdetection/focalloss.txt', "a")
    f.write(str(epochs) + '\t' + str(np.average(losses)) +'\n')
    f.close()
    del predicted_locs, predicted_classes_scores, images, boxes, labels


def save_checkpoint(epoch, model, optim, path):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optim}
    filename = path
    torch.save(state, filename)


def adjust_learning_rate(optimizer, scale):
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*scale
    print('Decaying learning rate')


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
