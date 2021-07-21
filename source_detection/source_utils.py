# +
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from evaluation import box_coord



def open_detector_bundle(path):
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


# -

def annotate(img, bbox, labels):
    class_labels = ('pointlike gaussian', 'diffuse gaussian', 'diamond', 'square', 'background')
    color_map = ('w', 'g', 'r', 'y','brown')
    img_size = img.shape[0]
    fig, ax2 = plt.subplots(1,1)
    for j in range(bbox.shape[0]):
        true_label = labels[j][0]
        color = color_map[labels[j][0]]
        trux, truy, truw, truh = box_coord(bbox[j],img_size)
        trurect = patches.Rectangle((trux, truy), truw, truh, linewidth=1, edgecolor=color, facecolor='none')
        ax2.text(trux,(truy+truh-7),true_label, color = 'k',fontsize=8,backgroundcolor = color)
        ax2.add_patch(trurect)
    ax2.imshow(img,cmap = 'gist_heat')


