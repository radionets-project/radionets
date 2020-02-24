import numpy as np
import astropy.coordinates as ac


class source():
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def to_ecef(self, val=None, prop=False):
        if prop is True:
            quant = ac.EarthLocation(self.lon_prop,
                                     self.lat_prop).to_geocentric()
        else:
            quant = ac.EarthLocation([self.lon], [self.lat]).to_geocentric()
        if val is not None:
            quant = ac.EarthLocation([val[0]], [val[1]]).to_geocentric()
        x = quant[0].value
        y = quant[1].value
        z = quant[2].value
        return x, y, z

    def propagate(self, num_steps=None, multi_pointing=False):
        if num_steps is None:
            num_steps = np.random.randint(30, 60)
        lon_start = self.lon
        lon_stop = lon_start - num_steps
        lon_step = 0.5
        lon = np.arange(lon_stop, lon_start, lon_step)
        lon = lon[::-1]

        lat_start = self.lat
        direction = np.sign(np.random.randint(0, 1) - 0.5)
        lat_stop = round((lat_start + direction * num_steps/50) + 0.005, 3)
        lat_step = 0.01
        if lat_start > lat_stop:
            lat = np.arange(lat_stop, lat_start, lat_step)
            lat = lat[::-1]
        else:
            lat = np.arange(lat_start, lat_stop, lat_step)

        if len(lon) != len(lat):
            raise ValueError('Length of lon and lat are different!')

        if multi_pointing is True:
            lon = self.mod_delete(lon, 5, 20)
            lat = self.mod_delete(lat, 5, 20)

        self.lon_prop = lon
        self.lat_prop = lat
        return lon, lat

    def mod_delete(self, a, n, m):
        return a[np.mod(np.arange(a.size), n + m) < n]


class antenna():
    def __init__(self, X, Y, Z):
        self.all = np.array(list(zip(X, Y, Z)))
        self.len = len(self.all)
        self.baselines = self.len * (self.len - 1)
        self.X = X
        self.Y = Y
        self.Z = Z
        self.to_geodetic(self.X, self.Y, self.Z)

    def to_geodetic(self, x_ref, y_ref, z_ref, enu=False):
        import astropy.units as u
        quant = ac.EarthLocation(x_ref, y_ref, z_ref, u.meter).to_geodetic()
        if enu is True:
            return quant.lon.deg, quant.lat.deg
        else:
            self.lon = quant.lon.deg
            self.lat = quant.lat.deg

    def get_baselines(self):
        x_base = ([])
        y_base = ([])
        for i in range(self.len):
            ref = np.ones((self.len, 2)) * ([self.x_enu[i], self.y_enu[i]])
            pairs = np.array([self.x_enu, self.y_enu])
            baselines = np.array(list(zip(ref, pairs.T))).ravel()
            x = baselines[0::2]
            y = baselines[1::2]
            x_base = np.append(x_base, x)
            y_base = np.append(y_base, y)
        return x_base, y_base

    def to_enu(self, x_ref, y_ref, z_ref):
        lon_ref, lat_ref = self.to_geodetic(x_ref, y_ref, z_ref, enu=True)
        ref = np.array(list(zip(x_ref, y_ref, z_ref)))

        def rot(lon, lat):
            lon = np.deg2rad(lon)
            lat = np.deg2rad(lat)
            return np.array([[-np.sin(lon), np.cos(lon), 0],
                             [-np.sin(lat)*np.cos(lon),
                             -np.sin(lat)*np.sin(lon), np.cos(lat)],
                             [np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)]
                             ])

        enu = np.array([rot(lon_ref[j], lat_ref[j]) @ (self.all[i] - ref[j])
                        for i in range(self.len) for j in range(len(lon_ref))])
        self.ant_enu = enu
        self.x_enu = enu.ravel()[0::3]
        self.y_enu = enu.ravel()[1::3]
        self.z_enu = enu.ravel()[2::3]
        return self.x_enu, self.y_enu

    def get_uv(self):
        u = ([])
        v = ([])
        steps = int(len(self.x_enu) / self.len)
        for j in range(steps):
            for i in range(self.len):
                x = self.x_enu[j::steps]
                y = self.y_enu[j::steps]
                x_ref = x[i] * np.ones(self.len)
                y_ref = y[i] * np.ones(self.len)
                x_base = x - x_ref
                y_base = y - y_ref
                x_base = x_base[x_base != 0] / 0.02
                y_base = -y_base[y_base != 0] / 0.02
                u = np.append(u, x_base)
                v = np.append(v, y_base)

        if len(u) != len(v):
            raise ValueError('Length of u and v are different!')
        return u, v, steps


def get_uv_coverage(source, antenna, iterate=False):
    antenna.to_enu(*source.to_ecef(prop=True))
    u, v, steps = antenna.get_uv()

    if iterate is True:
        num_base = antenna.baselines
        u.resize((steps, num_base))
        v.resize((steps, num_base))

    return u, v, steps


def create_mask(u, v, size=64):
    ''' Create 2d mask from a given (uv)-coverage

    u: array of u coordinates
    v: array of v coordinates
    size: number of bins
    '''
    uv_hist, _, _ = np.histogram2d(u, v, bins=size)
    # exclude center
    ex_l = size // 2 - 5
    ex_h = size // 2 + 5
    uv_hist[ex_l:ex_h, ex_l:ex_h] = 0
    mask = uv_hist > 0
    return np.rot90(mask)


def sample_freqs(img, ant_config_path, size=64, lon=None, lat=None,
                 num_steps=None, plot=False):
    ant = antenna(*get_antenna_config(ant_config_path))
    if lon is None:
        lon = np.random.randint(-90, -70)
    if lat is None:
        lat = np.random.randint(30, 80)
    s = source(lon, lat)
    s.propagate(num_steps=num_steps, multi_pointing=True)
    u, v, _ = get_uv_coverage(s, ant, iterate=False)
    # print(size)
    mask = create_mask(u, v, size)
    img = img.reshape(64,64)
    img[~mask] = 0
    img.reshape(4096)
    '''
    if mnist:
        img = img.reshape(64, 64)
        img[~mask] = 0
        img = img.reshape(4096)
    '''
    if plot is True:
        return img, mask
    else:
        return img


def get_antenna_config(config_path):
    config = config_path
    x, y, z, _, _ = np.genfromtxt(config, unpack=True)
    ant_pos = np.array([x, y, z])
    return ant_pos
