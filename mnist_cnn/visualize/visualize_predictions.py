from utils import load_model, load_pre_model, eval_model
from mnist_cnn.preprocessing import get_h5_data, prepare_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch

i = 12

model_path = '../models/simple_cnn.py'
pretrained_path = '../models/mnist_mixup_adam_leaky_2500.model'
path_valid = '../data/mnist_samp_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])
img = torch.tensor(x_valid[i])
img_reshaped = img.view(1, 1, 64, 64)
print(img_reshaped.shape)


model = load_model(model_path)
model_pre = load_pre_model(model, pretrained_path)
prediction = eval_model(img_reshaped, model_pre)

print(prediction)
print(prediction.shape)

pred_img = prediction.reshape(64, 64).numpy()
plt.imshow(pred_img)
plt.colorbar()
plt.show()


plt.imshow(y_valid[i].reshape(64, 64))
plt.colorbar()
plt.show()