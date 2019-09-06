import numpy as np
import matplotlib.pyplot as plt


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

def track_uv(listha, lengthbaseline, elevation, azimuth, latitude, dec, ntimeslots):

    UVW = np.zeros((ntimeslots, 3), dtype=float)
    for i in range(ntimeslots):
        UVW[i, :] = np.dot(xyz_to_baseline(listha[i], dec),baseline_to_xyz(lengthbaseline, azimuth, elevation, latitude)).T
    return UVW

def get_uv_tracks(antennaPos,L,dec,h,Ntimes,lamb):
    B = baseline_angles(antennaPos,lamb)
    na = len(antennaPos) #number of antennas
    nbl = round(na*(na-1)/2) #number of baselines
    uv_max = 0
    uv = np.array([track_uv(h,B[i, 0], 0., B[i, 1], L, dec, Ntimes)/1e6 for i in range(nbl)])
    if uv.max() > uv_max : uv_max=uv.max()
    return uv

def simulate_uv(ant_pos):
    # Observation parameters
    c = 3e8
    f = 15e9
    lam = c/f
    time_steps = 500
    Ntimes=10
    h = np.linspace(-6, 6,num=time_steps)*np.pi/12 # Hour angle window
    # declination convert in radian
    L = np.radians(30.6349)
    dec = np.radians(np.random.randint(0, 90))
    
    start = np.random.random(1)
    num_pointings = np.random.randint(6, 12)
    pointings = start + range(num_pointings)
    uv_tracks = np.array([get_uv_tracks(ant_pos, L, dec, h+i, Ntimes, lam) for i in pointings])
    return uv_tracks

def get_mask(uv_tracks):
    u_cords = uv_tracks[:,:,:,0].ravel()
    v_cords = uv_tracks[:,:,:,1].ravel()
    uv_hist, _, _ = np.histogram2d(u_cords ,v_cords , bins=64)
    
    # exclude center
    uv_hist[31,31] = 0
    uv_hist[31,32] = 0
    uv_hist[31,33] = 0
    uv_hist[32,31] = 0
    uv_hist[32,32] = 0
    uv_hist[32,33] = 0
    uv_hist[33,31] = 0
    uv_hist[33,32] = 0
    uv_hist[33,33] = 0

    mask = uv_hist > 0
    return mask

def plot_sampled_freq(mask, img):
    img[~mask] = 0
    plt.imshow(img, cmap='RdBu_r', vmin=-5, vmax=5)
    plt.colorbar()

def sample_freqs(img, ant_pos):
    img = img.reshape(64, 64)
    uv_tracks = simulate_uv(ant_pos)
    mask = get_mask(uv_tracks)
    img[~mask] = 0
    return img