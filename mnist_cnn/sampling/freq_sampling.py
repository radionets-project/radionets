# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from preprocessing import get_h5_data
from sampling import simulate_uv, get_mask, plot_sampled_freq
# -

x_mnist, y_mnist = get_h5_data('../data/mnist_valid.h5', columns=['x_valid', 'y_valid'])

config = 'vlba.txt'
x, y, z, dish, statio = np.genfromtxt(config, unpack=True)
ant_pos = np.array(list(zip(x, y)))

# +
uv_tracks = simulate_uv(ant_pos)
print(uv_tracks.shape)
mask = get_mask(uv_tracks)

img = np.log(x_mnist[0].reshape(64, 64))
plot_sampled_freq(mask, img)
# -



# +
# antennaPosition = np.array(list(zip(x, y)))
# mxabs = np.max(abs(antennaPosition[:]))*1.1;
# print(mxabs)
# # make use of pylab librery to plot
# plt.rcParams.update({'font.size': 20})
# fig=plt.figure(figsize=(6,6))
# plt.plot((antennaPosition[:,0]-np.mean(antennaPosition[:,0]))/1e3,          (antennaPosition[:,1]-np.mean(antennaPosition[:,1]))/1e3, 'o',
#          label='Antenna positions')
# plt.axes().set_aspect('equal')
# plt.xlim(-mxabs/1e3, mxabs/1e3)
# plt.ylim(-mxabs/1e3, (mxabs+5)/1e3)
# plt.legend()

# +
# def plotuv(antennaPos,L,dec,h,Ntimes,lamb):
# #     plt.clf()
#     plt.rcParams.update({'font.size': 20})
#     B = baseline_angles(antennaPos,lamb)
#     na = len(antennaPos) #number of antennas
#     nbl = round(na*(na-1)/2) #number of baselines
#     maxuv=0.
#     for i in range (nbl):
#         uv = track_uv(h,B[i, 0], 0., B[i, 1], L, dec, Ntimes)/1e6;
#         if uv.max() > maxuv : maxuv=uv.max()
#         plt.plot(uv[:,0], uv[:,1],ms=2.5, marker='.', color='#1f77b4')#, color='#1f77b4')
#         plt.plot(-uv[:,0], -uv[:,1],ms=2.5,marker='.', color='#1f77b4')#, color='#ff7f0e')
        
#     plt.plot([], [], linestyle='none', marker='.', markersize=3, label='V(u,v)')
#     plt.plot([], [], linestyle='none', marker='.', markersize=3, label='V(-u,-v)')
#     plt.xlabel('u / k$\lambda$')
#     plt.ylabel('v (klambda)')
#     #plt.title('uv coverage')
#     mb = maxuv*1.1 #5*np.sqrt((uv**2).sum(1)).max()
# #     plt.axes().set_aspect('equal')
# #     plt.xlim(-40e3,40e3)
# #     plt.ylim(-40e3,40e3)
# #     plt.legend(loc=2, markerscale=3)

# +
# Ntimes=10

# fig=plt.figure(figsize=(8,8))

# [plotuv(antennaPosition,L,dec,h+i,Ntimes,lam) for i in [0, 1, 2, 3, 4]]
# # plotuv(antennaPosition,L,dec,h+1,Ntimes,lam)
# plt.xlabel('u / k$\lambda$')
# plt.ylabel('v / k$\lambda$')
# # plt.xlim(-40,40)
# # plt.ylim(-40,40)
# # plt.yticks([-40, -20, 0, 20, 40])
# # plt.savefig('vlba_3steps.jpg', dpi=150, bbox_inches='tight', pad_inches=0)
