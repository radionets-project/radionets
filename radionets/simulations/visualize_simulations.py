import radionets.simulations.layouts.layouts as layouts
from radionets.simulations.uv_simulations import Antenna, Source
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt

from pathlib import Path

from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from radionets.evaluation.utils import make_axes_nice


def create_path(path):
    p = Path(path).parent
    p.mkdir(parents=True, exist_ok=True)


def vlba_basic(center_lon=-110, center_lat=27.75):
    layout = getattr(layouts, "vlba")
    ant = Antenna(*layout())
    ant.to_geodetic(ant.X, ant.Y, ant.Z)

    s = Source(center_lon, center_lat)
    s.to_ecef(prop=False)

    ant_lon = ant.lon
    ant_lat = ant.lat

    ant.to_enu(*s.to_ecef(prop=False))
    base_lon, base_lat = ant.get_baselines()
    return ant_lon, ant_lat, base_lon, base_lat


def plot_vlba(
    out_path, ant_lon, ant_lat, base_lon, base_lat, center_lon=-110, center_lat=27.75
):
    extent = [-155, -65, 10, 45.5]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    stamen_terrain = cimgt.Stamen("terrain-background")

    plt.figure(figsize=(5.78 * 2, 3.57))
    ax = plt.axes(projection=ccrs.Orthographic(central_lon, central_lat))
    ax.set_extent(extent)

    ax.plot(
        ant_lon,
        ant_lat,
        marker=".",
        color="black",
        linestyle="none",
        markersize=6,
        zorder=10,
        transform=ccrs.Geodetic(),
        label="Antenna positions",
    )
    ax.plot(
        base_lon,
        base_lat,
        zorder=5,
        linestyle="-",
        linewidth=0.5,
        alpha=0.7,
        color="#d62728",
        label="Baselines",
    )

    ax.add_image(stamen_terrain, 4)

    leg = plt.legend(markerscale=1.5, fontsize=7, loc=2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.05)


def create_vlba_overview(out_path):
    ant_lon, ant_lat, base_lon, base_lat = vlba_basic()
    create_path(out_path)
    plot_vlba(out_path, ant_lon, ant_lat, base_lon, base_lat)


def plot_source(img, log=False, out_path=None):
    fig = plt.figure(figsize=(5.78, 3.57))
    ax = fig.add_subplot(111)
    img = np.abs(img)
    ax.set_xlabel("l", fontsize=9)
    ax.set_ylabel("m", fontsize=9)
    if log is True:
        s = ax.imshow(img, cmap="inferno", norm=LogNorm(vmin=1e-8, vmax=img.max()))
    else:
        s = ax.imshow(img, cmap="inferno")
    make_axes_nice(fig, ax, s, "")

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.05)


def plot(img, ax, phase=False):
    if phase:
        im = ax.imshow(img, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
    else:
        im = ax.imshow(img, cmap="inferno", norm=LogNorm())
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    return im


def plot_spectrum(img, out_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.78, 3.57 * 2))
    amp = np.abs(img)
    phase = np.angle(img)

    im1 = plot(amp, ax1)
    make_axes_nice(fig, ax1, im1, "")
    ax1.set_xlabel("u", fontsize=9)
    ax1.set_ylabel("v", fontsize=9)

    im2 = plot(phase, ax2, phase=True)
    make_axes_nice(fig, ax2, im2, "", phase=True)
    ax2.set_xlabel("u", fontsize=9)
    ax2.set_ylabel("v", fontsize=9)

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.05)


def ft(img):
    return np.fft.fftshift(np.fft.fft2(img))


def plot_uv_coverage(u, v, ax):
    ax.plot(u, v, marker="o", linestyle="none", markersize=2, color="#1f77b4")
    ax.set_xlabel(r"u / $\lambda$", fontsize=9)
    ax.set_ylabel(r"v / $\lambda$", fontsize=9)


def plot_vlba_uv(u, v, out_path=None):
    fig, ax = plt.subplots(1, figsize=(3.57, 3.57), dpi=100)
    plot_uv_coverage(u, v, ax)
    ax.set_ylim(-5e8, 5e8)
    ax.set_xlim(-5e8, 5e8)
    plt.tick_params(axis="both", labelsize=9)

    ax.axis("equal")
    fig.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.05)


def plot_baselines(antenna):
    x_base, y_base = antenna.get_baselines()
    plt.plot(
        x_base,
        y_base,
        linestyle="--",
        color="#2ca02c",
        zorder=0,
        label="Baselines",
        alpha=0.35,
    )


def plot_antenna_distribution(
    source_lon,
    source_lat,
    source,
    antenna,
    baselines=False,
    end=False,
    lon_start=None,
    lat_start=None,
    out_path=None,
):
    x, y, z = source.to_ecef(val=[source_lon, source_lat])  # only use source ?
    x_enu_ant, y_enu_ant = antenna.to_enu(x, y, z)

    plt.figure(figsize=(5.78, 3.57), dpi=100)
    ax = plt.axes(projection=ccrs.Orthographic(source_lon, source_lat))
    ax.set_global()
    ax.coastlines()

    plt.plot(
        x_enu_ant,
        y_enu_ant,
        marker="o",
        markersize=3,
        color="#1f77b4",
        linestyle="none",
        label="Antenna positions",
    )
    print(x)
    print(y)
    plt.plot(
        x,
        y,
        marker="*",
        linestyle="none",
        color="#ff7f0e",
        markersize=10,
        transform=ccrs.Geodetic(),
        zorder=10,
        label="Projected source",
    )

    if baselines:
        plot_baselines(antenna)  # projected baselines

    if end:
        x_start, y_start, _ = source.to_ecef(val=[lon_start, lat_start])

        print(x_start)
        print(y_start)
        ax.plot(
            np.array([x, x_start]),
            np.array([y, y_start]),
            marker=".",
            linestyle="--",
            color="#d62728",
            linewidth=1,
            transform=ccrs.Geodetic(),
            zorder=10,
            label="Source path",
        )
        ax.plot(
            x_start,
            y_start,
            marker=".",
            color="green",
            zorder=10,
            label="hi",
            transform=ccrs.Geodetic(),
        )

    plt.legend(
        fontsize=9, markerscale=1.5, bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.0
    )

    if out_path is not None:
        plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.05)


def plot_mask(mask):
    fig = plt.figure(figsize=(5.78, 3.57), dpi=100)
    ax = fig.add_subplot(111)

    im = ax.imshow(mask, cmap="inferno")
    values = np.unique(mask.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    names = ["Unsampled", "Sampled"]
    patches = [
        mpatches.Patch(color=colors[i], label=f"{names[i]}") for i in range(len(values))
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_xlabel("u", fontsize=9)
    ax.set_ylabel("v", fontsize=9)
    plt.tight_layout()


def apply_mask(img, mask):
    img = img.copy()
    img[~mask.astype(bool)] = 0
    return img
