# +
import h5py
import numpy as np

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
