from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

__all__ = ["Hist"]


class Hist:
    def __init__(
        self,
        outpath,
        plot_format: str = "png",
        hist_kwargs: dict | None = None,
        save_kwargs: dict | None = None,
    ):
        self.outpath = outpath
        self.plot_format = plot_format

        if not Path(self.outpath).exists():
            Path(self.outpath).mkdir(parents=True, exist_ok=True)

        self.hist_kwargs = hist_kwargs
        if not self.hist_kwargs:
            self.hist_kwargs = dict(
                color="darkorange",
                linewidth=3,
                histtype="step",
                alpha=0.75,
            )

        self.save_kwargs = save_kwargs
        if not self.save_kwargs:
            self.save_kwargs = dict(
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=150,
            )

    def _preproc_vals(
        self, vals: torch.Tensor | np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        if torch.is_tensor(vals):
            vals = vals.numpy()

        mean = vals.mean()
        # NOTE: passing the mean to std() prevents its recalculation
        std = vals.std(ddof=1, mean=mean)

        return vals, mean, std

    def _add_mean_std_text(self, ax: plt.axes, mean: float, std: float):
        ax.text(
            0.1,
            0.8,
            f"Mean: {mean:.2f}\nStd: {std:.2f}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.8,
            ),
        )

    def _get_rect_patch(self) -> Rectangle:
        kwargs = dict(
            width=1,
            height=1,
            fc="w",
            fill=False,
            edgecolor="#1f77b4",
            linewidth=2,
        )

        rect_1 = Rectangle((0, 0), **kwargs)
        rect_2 = Rectangle((0, 0), **kwargs)
        return rect_1, rect_2

    def area(
        self,
        vals: torch.tensor,
        bins: int = 30,
        return_fig: bool = False,
    ):
        vals, mean, std = self._preproc_vals(vals)

        fig, ax = plt.subplots(1, figsize=(6, 4))

        ax.hist(
            vals,
            bins=bins,
            **self.hist_kwargs,
        )
        ax.axvline(1, color="red", linestyle="dashed")
        ax.set(xlabel="Ratio of areas", ylabel="Number of sources")

        self._add_mean_std_text(ax, mean, std)

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/hist_area.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def dynamic_ranges(
        self,
        dr_truth: torch.tensor,
        dr_pred: torch.tensor,
        return_fig: bool = False,
    ):
        fig, ax = plt.subplots(2, 1, figsize=(6, 12), layout="constrained")
        ax[0].hist(dr_truth, 51, **self.hist_kwargs)
        ax[0].set(
            title="True Images",
            xlabel="Dynamic range",
            ylabel="Number of sources",
        )

        ax[1].hist(dr_pred, 25, **self.hist_kwargs)
        ax[1].set(
            title="Predictions",
            xlabel="Dynamic range",
            ylabel="Number of sources",
        )

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/dynamic_ranges.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def gan_sources(
        self,
        ratio,
        num_zero,
        above_zero,
        below_zero,
        num_images,
    ):
        bins = np.arange(0, ratio.max() + 0.1, 0.1)

        fig, ax = plt.subplots(1, layout="constrained")
        ax.hist(
            ratio,
            bins=bins,
            histtype="step",
            label=f"mean: {ratio.mean():.2f}, max: {ratio.max():.2f}",
        )
        ax.set(
            xlabel=r"Maximum difference to maximum true flux ratio",
            ylabel=r"Number of sources",
        )
        ax.legend(loc="best")

        outpath = str(self.outpath) + f"/ratio.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

        plt.close(fig)

        fig, ax = plt.subplots(1, layout="constrained")
        bins = np.arange(0, 102, 2)
        num_zero = num_zero.reshape(4, num_images)

        for i, label in enumerate(["1e-4", "1e-3", "1e-2", "1e-1"]):
            ax.hist(num_zero[i], bins=bins, histtype="step", label=label)

        ax.set(
            xlabel=r"Proportion of pixels close to 0 / %",
            ylabel=r"Number of sources",
        )
        ax.legend(loc="upper center")

        outpath = str(self.outpath) + f"/num_zeros.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

        plt.close(fig)

        fig, ax = plt.subplots(1, layout="constrained")
        bins = np.arange(0, 102, 2)
        ax.hist(
            above_zero,
            bins=bins,
            histtype="step",
            label=f"Above, mean: {above_zero.mean():.2f}%, max: {above_zero.max():.2f}%",  # noqa: E501
        )
        ax.hist(
            below_zero,
            bins=bins,
            histtype="step",
            label=f"Below, mean: {below_zero.mean():.2f}%, max: {below_zero.max():.2f}%",  # noqa: E501
        )
        ax.set(
            xlabel=r"Proportion of pixels below or above 0%",
            ylabel=r"Number of sources",
        )
        ax.legend(loc="upper center")

        outpath = str(self.outpath) + f"/above_below.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def jet_angles(self, vals: torch.tensor, return_fig: bool = False):
        vals, mean, std = self._preproc_vals(vals)

        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        ax[0].hist(vals, 51, **self.hist_kwargs)
        ax[0].set(
            xlabel="Offset / deg",
            ylabel="Number of sources",
        )

        extra_1, extra_2 = self._get_rect_patch()
        ax[0].legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

        ax[1].hist(vals[(vals > -10) & (vals < 10)], 25, **self.hist_kwargs)
        ax[1].set(
            xticks=[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10],
            xlabel="Offset / deg",
            ylabel="Number of sources",
        )

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/jet_offsets.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def jet_gaussian_distance(self, dist: torch.tensor, return_fig: bool = False):
        """
        Plotting the distances between predicted and true component of several images.
        Parameters
        ----------
        dist: 2d array
            array of shape (n, 2), where n is the number of distances
        """

        ran = [0, 50]

        fig, ax = plt.subplots(1, layout="constrained")

        for i in range(10):
            ax.hist(
                dist[dist[:, 0] == i][:, 1],
                bins=20,
                range=ran,
                alpha=0.7,
                label=f"Component {i}",
            )

        ax.set(xlabel="Distance", ylabel="Counts")
        ax.legend()

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/hist_jet_gaussian_distance.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def mean_diff(self, vals: torch.tensor, return_fig: bool = False):
        vals, mean, std = self._preproc_vals(vals)

        fig, ax = plt.subplots(1, figsize=(6, 4))

        ax.hist(vals, 51, **self.hist_kwargs)
        ax.set(
            xlabel="Mean flux deviation / %",
            ylabel="Number of sources",
        )

        extra_1, extra_2 = self._get_rect_patch()
        ax.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/mean_diff.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def ms_ssim(
        self,
        vals: torch.tensor,
        bins: int = 30,
        return_fig: bool = False,
    ):
        vals, mean, std = self._preproc_vals(vals)

        fig, ax = plt.subplots(1, figsize=(6, 4), layout="constrained")
        ax.hist(vals, bins=bins, **self.hist_kwargs)
        ax.set(
            xlabel="ms ssim",
            ylabel="Number of sources",
        )

        self._add_mean_std_text(ax, mean, std)

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/ms_ssim.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def peak_intensity(
        self,
        vals: torch.tensor,
        bins: int = 30,
        return_fig: bool = False,
    ):
        vals, mean, std = self._preproc_vals(vals)

        fig, ax = plt.subplots(1, figsize=(6, 4), layout="constrained")

        ax.hist(vals, bins=bins, **self.hist_kwargs)
        ax.axvline(1, color="red", linestyle="dashed")
        ax.set(
            xlabel="Ratio of peak flux densities",
            ylabel="Number of sources",
        )

        self._add_mean_std_text(ax, mean, std)

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/intensity_peak.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def point(self, vals: torch.tensor, mask: torch.tensor, return_fig: bool = False):
        binwidth = 5
        min_all = vals.min()
        bins = np.arange(min_all, 100 + binwidth, binwidth)

        mean_point = np.mean(vals[mask])
        std_point = np.std(vals[mask], ddof=1)
        mean_extent = np.mean(vals[~mask])
        std_extent = np.std(vals[~mask], ddof=1)

        fig, ax = plt.subplots(1, figsize=(6, 4), layout="constrained")

        ax.hist(vals[mask], bins=bins, **self.hist_kwargs)
        ax.hist(vals[~mask], bins=bins, **self.hist_kwargs)

        ax.axvline(0, linestyle="dotted", color="red")
        ax.set(
            xlabel="Mean specific intensity deviation",
            ylabel="Number of sources",
        )

        extra_1, extra_2 = self._get_rect_patch()
        ax.legend(
            [extra_1, extra_2],
            [
                rf"Point: $({mean_point:.2f}\pm{std_point:.2f})\,\%$",
                rf"Extended: $({mean_extent:.2f}\pm{std_extent:.2f})\,\%$",
            ],
        )

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/hist_point.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def sum_intensity(
        self,
        vals: torch.tensor,
        bins: int = 30,
        return_fig: bool = False,
    ):
        vals, mean, std = self._preproc_vals(vals)

        fig, ax = plt.subplots(1, figsize=(6, 4), layout="constrained")

        ax.hist(vals, bins=bins, **self.hist_kwargs)
        ax.axvline(1, color="red", linestyle="dashed")
        ax.set(
            xlabel="Ratio of integrated flux densities",
            ylabel="Number of sources",
        )

        self._add_mean_std_text(ax, mean, std)

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/intensity_sum.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)

    def unc(self, vals: torch.tensor, return_fig: bool = False):
        vals, mean, std = self._preproc_vals(vals)

        bins = np.arange(0, 105, 5)

        fig, ax = plt.subplots(1, figsize=(6, 4), layout="constrained")

        ax.hist(vals, bins=bins, **self.hist_kwargs)
        ax.set(
            xlabel="Percentage of matching pixels",
            ylabel="Number of sources",
        )

        self._add_mean_std_text(ax, mean, std)

        if return_fig:
            return fig, ax

        outpath = str(self.outpath) + f"/hist_unc.{self.plot_format}"
        fig.savefig(outpath, **self.save_kwargs)
