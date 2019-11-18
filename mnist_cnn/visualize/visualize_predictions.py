from mnist_cnn.visualize.utils import (load_architecture, load_pre_model,
                                       eval_model)
from mnist_cnn.preprocessing import get_h5_data
from mnist_cnn.inspection import get_normalization
import matplotlib.pyplot as plt
import torch

i = 0

arch_path = '../models/cnn_architecture.py'
pretrained_path = '../models/hamburg2.model'
path_valid = '../data/mnist_samp_valid.h5'
norm_path = '../data/normalization_factors.csv'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])
img = torch.tensor(x_valid[i])
img_log = torch.log(img)
img_reshaped = img_log.view(1, 1, 64, 64)
print(img_reshaped.shape)
img_normed = get_normalization(img_reshaped, norm_path)
print(img_normed.shape)

model = load_architecture(arch_path)
model_pre = load_pre_model(model, pretrained_path)
prediction = eval_model(img_normed, model_pre)

print(prediction)
print(prediction.shape)

pred_img = prediction.reshape(64, 64).numpy()
plt.imshow(pred_img)
plt.colorbar()
plt.show()


plt.imshow(y_valid[i].reshape(64, 64))
plt.colorbar()
plt.show()
