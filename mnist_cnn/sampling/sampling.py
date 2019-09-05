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

from preprocessing import get_h5_data
import numpy as np
import matplotlib.pyplot as plt
# -

x, y = get_h5_data('../data/mnist_valid.h5', columns=['x_valid', 'y_valid'])

x.shape

# +
config = 'vlba.txt'

x, y, z, dish, statio = np.genfromtxt(config, unpack=True)
# -

antennaPosition = np.array(list(zip(x, y)))
antennaPosition[:,1]

antennaPosition = np.array(list(zip(x, y)))
mxabs = np.max(abs(antennaPosition[:]))*1.1;
print(mxabs)
# make use of pylab librery to plot
plt.rcParams.update({'font.size': 20})
fig=plt.figure(figsize=(6,6))
plt.plot((antennaPosition[:,0]-np.mean(antennaPosition[:,0]))/1e3,          (antennaPosition[:,1]-np.mean(antennaPosition[:,1]))/1e3, 'o',
         label='Antenna positions')
plt.axes().set_aspect('equal')
plt.xlim(-mxabs/1e3, mxabs/1e3)
plt.ylim(-mxabs/1e3, (mxabs+5)/1e3)
plt.legend()


def baseline_angles(ant_pos,lamb):
    # number of antennas
    na = len(ant_pos)
    # number of independent baselines
    nbl = round(na*(na-1)/2)
    length_angle = np.zeros((nbl, 2))
    k = 0
    for i in range(na):
        for j in range(i+1, na):
            length_angle[k,0] = lamb**(-1)*np.sqrt((ant_pos[i,0]-ant_pos[j,0])**2 + (ant_pos[i,1]-ant_pos[j,1])**2)
            length_angle[k,1] = np.arctan2((ant_pos[i,1]-ant_pos[j,1]) , (ant_pos[i,0]-ant_pos[j,0]))
            k = k +1
    return length_angle


baseline_angles(antennaPosition, 0.02)


def xyz_to_baseline(ha, dec):
    a1 = np.sin(ha)
    a2 = np.cos(ha)
    a3 = 0.

    b1 = -1*np.sin(dec)*np.cos(ha)
    b2 = np.sin(dec)*np.sin(ha)
    b3 = np.cos(dec)

    c1 = np.cos(dec)*np.cos(ha)
    c2 = -1*np.cos(dec)*np.sin(ha)
    c3 = np.sin(dec)

    return np.array([(a1,a2,a3),(b1,b2,b3),(c1,c2,c3)])


def baseline_to_xyz(lengthbaseline, elevation, azimuth, latitude):

    x = np.cos(latitude)*np.sin(elevation) - np.sin(latitude)*np.cos(elevation)*np.cos(azimuth)

    y = np.cos(elevation)*np.sin(azimuth)

    z = np.sin(latitude)*np.sin(elevation) + np.cos(latitude)*np.cos(elevation)*np.cos(azimuth)

    xyz = np.array([(x,y,z)])

    return lengthbaseline * xyz.T


def plotuv(antennaPos,L,dec,h,Ntimes,lamb):
#     plt.clf()
    plt.rcParams.update({'font.size': 20})
    B = baseline_angles(antennaPos,lamb)
    na = len(antennaPos) #number of antennas
    nbl = round(na*(na-1)/2) #number of baselines
    maxuv=0.
    for i in range (nbl):
        uv = track_uv(h,B[i, 0], 0., B[i, 1], L, dec, Ntimes)/1e6;
        if uv.max() > maxuv : maxuv=uv.max()
        plt.plot(uv[:,0], uv[:,1],ms=2.5, marker='.', color='#1f77b4')#, color='#1f77b4')
        plt.plot(-uv[:,0], -uv[:,1],ms=2.5,marker='.', color='#1f77b4')#, color='#ff7f0e')
        
    plt.plot([], [], linestyle='none', marker='.', markersize=3, label='V(u,v)')
    plt.plot([], [], linestyle='none', marker='.', markersize=3, label='V(-u,-v)')
    plt.xlabel('u / k$\lambda$')
    plt.ylabel('v (klambda)')
    #plt.title('uv coverage')
    mb = maxuv*1.1 #5*np.sqrt((uv**2).sum(1)).max()
#     plt.axes().set_aspect('equal')
#     plt.xlim(-40e3,40e3)
#     plt.ylim(-40e3,40e3)
#     plt.legend(loc=2, markerscale=3)


def track_uv(listha, lengthbaseline, elevation, azimuth, latitude, dec, ntimeslots):

    UVW = np.zeros((ntimeslots, 3), dtype=float)
    for i in range(ntimeslots):
        UVW[i, :] = np.dot(xyz_to_baseline(listha[i], dec),baseline_to_xyz(lengthbaseline, azimuth, elevation, latitude)).T
    return UVW


def get_uv_tracks(antennaPos,L,dec,h,Ntimes,lamb):
    B = baseline_angles(antennaPos,lamb)
    na = len(antennaPos) #number of antennas
    nbl = round(na*(na-1)/2) #number of baselines
    maxuv=0.
    print(B)
    for i in range (nbl):
        uv = track_uv(h,B[i, 0], 0., B[i, 1], L, dec, Ntimes)/1e6;
        if uv.max() > maxuv : maxuv=uv.max()
    print(uv[0])


[get_uv_tracks(antennaPosition,L,dec,h+i,Ntimes,lam) for i in [0]]

# +
# Observation parameters
c=3e8                                         # Speed of light
f=15e9                                      # Frequency
lam = c/f                                     # Wavelength 
print('wavelength:', lam, 'm')

time_steps = 500                              # time steps
h = np.linspace(-6, 6,num=time_steps)*np.pi/12 # Hour angle window

# declination convert in radian

L = np.radians(30.6349)      # Latitude of the VLA
dec = np.radians(30.)

# +
Ntimes=10

fig=plt.figure(figsize=(8,8))

[plotuv(antennaPosition,L,dec,h+i,Ntimes,lam) for i in [0, 2, 4, 6, 8]]
# plotuv(antennaPosition,L,dec,h+1,Ntimes,lam)
plt.xlabel('u / k$\lambda$')
plt.ylabel('v / k$\lambda$')
# plt.xlim(-40,40)
# plt.ylim(-40,40)
# plt.yticks([-40, -20, 0, 20, 40])
plt.savefig('vlba_3steps.jpg', dpi=150, bbox_inches='tight', pad_inches=0)
# -






