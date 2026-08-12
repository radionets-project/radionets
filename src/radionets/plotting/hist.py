import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radionets.core.logging import _setup_logger

__all__ = ["Hist"]


LOGGER = _setup_logger(namespace=__name__)


def stacked_label(model, val1_u, val2_u, val1_l, val2_l):
    label = rf"{model}, \textcolor{{spaceblack!70!white}}"
    label += rf"{{$\mu = \Vectorstack[l]{{\num{{{val1_u:.3f} \pm {val2_u:.3f}}}"
    label += rf" \textcolor{{spaceblack!50!white}}{{\num{{{val1_l:.3f} \pm {val2_l:.3f}}}}}}}$}}"  # noqa: E501

    return label


def hist_label(model, mean, std):
    label = rf"{model}, \textcolor{{spaceblack!60!white}}"
    label += rf"{{$\mu = \num{{{mean:.3f} \pm {std:.3f}}}$}}"

    return label


def set_label(model, mean_full, std_full, mean, std, stacked_label):
    if stacked_label:
        return stacked_label(model, mean_full, std_full, mean, std)

    return hist_label(model, mean, std)


class Hist:
    spaceblack: str = "#1a1a1a"

    def __init__(self, config):
        self.config = config

        self.models = config.general.display_names
        self.colors = config.general.colors

        if not self.models:
            # Model display names as generic A, B, C, and D
            self.models = "ABCD"[: len(config.paths.data_paths)]

        if not self.colors:
            self.colors: list = cmr.get_sub_cmap("inferno", 0.25, 0.8, N=4).colors  # ty:ignore[invalid-assignment]

            if len(self.models) == 1:
                self.colors = [self.spaceblack]

    def peak_flux(self):
        peak_cfg = self.config.peak_flux

        fig, ax = plt.subplots(**peak_cfg.fig.subplots.model_dump())

        for data_path, model, c in zip(
            self.config.paths.data_paths, self.models, self.colors
        ):
            data = pd.read_csv(data_path / "intensity.csv")["peak_flux"]
            self._hist(ax=ax, data=data, model=model, color=c, config=peak_cfg)

        _, ax_max = ax.get_ylim()

        outliers = False
        for i, (data_path, model) in enumerate(
            zip(self.config.paths.data_paths, self.models)
        ):
            data = pd.read_csv(data_path / "intensity.csv")["peak_flux"]

            if sum(data > peak_cfg.upper_bound) > 0:
                self._add_annotations(
                    i,
                    model,
                    ax=ax,
                    ax_max=ax_max,
                    condition=data > peak_cfg.upper_bound,
                )
                outliers = True

        if outliers:
            self._add_cutoff_line(peak_cfg.upper_bound, ax=ax, ax_max=ax_max)

        fig.legend(
            frameon=peak_cfg.legend.frameon,
            loc=peak_cfg.legend.loc,
            ncols=peak_cfg.legend.ncols,
            bbox_to_anchor=(peak_cfg.legend.legx, peak_cfg.legend.legy),
        )

        ax.set(
            xlabel="Ratio of Peak Flux Densities",
            ylabel="Number of Sources",
        )

        out_path = (
            self.config.paths.save_path
            / f"intensity_peak.{self.config.general.file_format}"
        )
        fig.savefig(out_path, **peak_cfg.fig.save.model_dump())

    def integrated_flux(self):
        int_cfg = self.config.integrated_flux

        fig, ax = plt.subplots(**int_cfg.fig.subplots.model_dump())

        for data_path, model, c in zip(
            self.config.paths.data_paths, self.models, self.colors
        ):
            data = pd.read_csv(data_path / "intensity.csv")["integrated_flux"]
            self._hist(ax=ax, data=data, model=model, color=c, config=int_cfg)

        _, ax_max = ax.get_ylim()

        outliers = False
        for i, (data_path, model) in enumerate(
            zip(self.config.paths.data_paths, self.models)
        ):
            data = pd.read_csv(data_path / "intensity.csv")["integrated_flux"]
            if sum(data > int_cfg.upper_bound) > 0:
                self._add_annotations(
                    i, model, ax, ax_max=ax_max, condition=data > int_cfg.upper_bound
                )

                outliers = True

        if outliers:
            self._add_cutoff_line(int_cfg.upper_bound, ax=ax, ax_max=ax_max)

        fig.legend(
            frameon=int_cfg.legend.frameon,
            loc=int_cfg.legend.loc,
            ncols=int_cfg.legend.ncols,
            bbox_to_anchor=(int_cfg.legend.legx, int_cfg.legend.legy),
        )

        ax.set(
            xlabel="Ratio of Integrated Flux Densities",
            ylabel="Number of Sources",
        )

        out_path = (
            self.config.paths.save_path
            / f"intensity_sum.{self.config.general.file_format}"
        )
        fig.savefig(out_path, **int_cfg.fig.save.model_dump())

    def angle(self):
        angle_cfg = self.config.angle

        fig, ax = plt.subplots(1, 2, **angle_cfg.fig.subplots.model_dump())

        for data_path, model, c in zip(
            self.config.paths.data_paths, self.models, self.colors
        ):
            data = pd.read_csv(data_path / "viewing_angle.csv")["diff"]
            LOGGER.info(
                f"{model}: {sum(data < angle_cfg.lower_bounds[0]) = } of {len(data)}"
            )
            LOGGER.info(
                f"{model}: {sum(data > angle_cfg.upper_bounds[0]) = } of {len(data)}"
            )

            mean = data.mean()
            std = data.std()

            ax[0].hist(
                data,
                bins=angle_cfg.hist0.bins,
                histtype=angle_cfg.hist0.histtype,
                color=c,
                range=(angle_cfg.lower_bounds[0], angle_cfg.upper_bounds[0]),
                label=hist_label(model, mean, std),
                linewidth=angle_cfg.hist0.linewidth,
            )

            LOGGER.info(
                f"{model}: {sum(data < angle_cfg.lower_bounds[1]) = } of {len(data)}"
            )
            LOGGER.info(
                f"{model}: {sum(data > angle_cfg.upper_bounds[1]) = } of {len(data)}"
            )

            ax[1].hist(
                data,
                bins=angle_cfg.hist1.bins,
                histtype=angle_cfg.hist1.histtype,
                color=c,
                range=(angle_cfg.lower_bounds[1], angle_cfg.upper_bounds[1]),
                linewidth=angle_cfg.hist1.linewidth,
            )

        fig.legend(
            frameon=angle_cfg.legend.frameon,
            loc=angle_cfg.legend.loc,
            ncols=angle_cfg.legend.ncols,
            bbox_to_anchor=(angle_cfg.legend.legx, angle_cfg.legend.legy),
        )

        for axs in ax:
            axs.set(
                xlabel=r"$\text{Jet Angle Offset} \mathbin{/} \unit{\deg}$",
                ylabel="Number of Sources",
            )

        out_path = (
            self.config.paths.save_path
            / f"viewing_angle.{self.config.general.file_format}"
        )
        fig.savefig(out_path, **angle_cfg.fig.save.model_dump())

    def mean_diff(self):
        md_cfg = self.config.mean_diff

        fig, ax = plt.subplots(**md_cfg.fig.subplots.model_dump())

        outliers_l = []
        outliers_r = []

        for data_path, model, c in zip(
            self.config.paths.data_paths, self.models, self.colors
        ):
            data = pd.read_csv(data_path / "mean_diff.csv")["mean_diff"]

            LOGGER.info(f"{model}: {sum(data < md_cfg.lower_bound) = } of {len(data)}")
            LOGGER.info(f"{model}: {sum(data > md_cfg.upper_bound) = } of {len(data)}")

            outliers_l.append(sum(data < md_cfg.lower_bound))
            outliers_r.append(sum(data > md_cfg.upper_bound))

            no_outliers = data[
                ~np.logical_or(data < md_cfg.lower_bound, data > md_cfg.upper_bound)
            ]
            mean = no_outliers.mean()
            std = no_outliers.std()

            mean_full = data.mean()
            std_full = data.std()

            ax.hist(
                data,
                bins=md_cfg.hist.bins,
                histtype=md_cfg.hist.histtype,
                color=c,
                range=(md_cfg.lower_bound, md_cfg.upper_bound),
                label=set_label(
                    model, mean_full, std_full, mean, std, md_cfg.hist.stacked_label
                ),
                linewidth=md_cfg.hist.linewidth,
            )

        _, ax_max = ax.get_ylim()

        for i, (data_path, model) in enumerate(
            zip(self.config.paths.data_paths, self.models)
        ):
            data = pd.read_csv(data_path / "mean_diff.csv")["mean_diff"]

            if sum(data > md_cfg.upper_bound) > 0:
                self._add_annotations(
                    i,
                    model,
                    ax,
                    ax_max=ax_max,
                    condition=data > md_cfg.upper_bound,
                    x1=md_cfg.upper_bound,
                    x2=40,
                )

            if sum(data < md_cfg.lower_bound) > 0:
                self._add_annotations(
                    i,
                    model,
                    ax,
                    ax_max=ax_max,
                    condition=data < md_cfg.lower_bound,
                    x1=md_cfg.lower_bound,
                    x2=-40,
                    ha="left",
                )

        if sum(outliers_r) > 0:
            self._add_cutoff_line(
                md_cfg.lower_bound,
                ax=ax,
                ax_max=ax_max,
                ha="left",
                outlier_scale=0.98,
                cutoff_scale=1.07,
            )

        if sum(outliers_l) > 0:
            self._add_cutoff_line(md_cfg.upper_bound, ax=ax, ax_max=ax_max)

        fig.legend(
            frameon=md_cfg.legend.frameon,
            loc=md_cfg.legend.loc,
            ncols=md_cfg.legend.ncols,
            bbox_to_anchor=(md_cfg.legend.legx, md_cfg.legend.legy),
        )

        ax.set(
            xlabel=r"$\text{Mean Flux Deviation} \mathbin{/} \unit{\percent}$",
            ylabel="Number of Sources",
        )

        out_path = (
            self.config.paths.save_path / f"mean_diff.{self.config.general.file_format}"
        )
        fig.savefig(out_path, **md_cfg.fig.save.model_dump())

    def area(self):
        area_cfg = self.config.area

        fig, ax = plt.subplots(**area_cfg.fig.subplots.model_dump())

        for data_path, model, c in zip(
            self.config.paths.data_paths, self.models, self.colors
        ):
            data = pd.read_csv(data_path / "area.csv")["source_area"]
            self._hist(ax=ax, data=data, model=model, color=c, config=area_cfg)

        _, ax_max = ax.get_ylim()

        for i, (data_path, model) in enumerate(
            zip(self.config.paths.data_paths, self.models)
        ):
            data = pd.read_csv(data_path / "area.csv")["source_area"]

            self._add_annotations(
                i, model, ax, ax_max=ax_max, condition=data > area_cfg.upper_bound
            )

        self._add_cutoff_line(area_cfg.upper_bound, ax=ax, ax_max=ax_max)

        fig.legend(
            frameon=area_cfg.legend.frameon,
            loc=area_cfg.legend.loc,
            ncols=area_cfg.legend.ncols,
            bbox_to_anchor=(area_cfg.legend.legx, area_cfg.legend.legy),
        )
        ax.set(
            xlabel=r"Ratio of Source Areas",
            ylabel="Number of Sources",
        )

        out_path = (
            self.config.paths.save_path
            / f"source_area_ratio.{self.config.general.file_format}"
        )
        fig.savefig(out_path, **area_cfg.fig.save.model_dump())

    def _hist(self, ax, data, model, color, config):
        LOGGER.info(f"{model}: {sum(data < config.lower_bound) = } of {len(data)}")
        LOGGER.info(f"{model}: {sum(data > config.upper_bound) = } of {len(data)}")

        mean = data[data <= config.upper_bound].mean()
        std = data[data <= config.upper_bound].std()

        mean_full = data.mean()
        std_full = data.std()

        ax.hist(
            data,
            bins=config.hist.bins,
            histtype=config.hist.histtype,
            color=color,
            range=(config.lower_bound, config.upper_bound),
            label=set_label(
                model, mean_full, std_full, mean, std, config.hist.stacked_label
            ),
            linewidth=config.hist.linewidth,
        )

    def _add_annotations(
        self,
        i,
        model,
        ax,
        ax_max,
        condition,
        x1=2,
        x2=1.85,
        frac=0.1,
        ha="right",
        yscale=1.0,
    ):
        if sum(condition) > 0:
            ax.annotate(
                rf"{model}: \num{{{sum(condition)}}}",
                (x1, yscale * ax_max - (i + 1 + frac) * frac * ax_max),
                xycoords="data",
                va="center",
                ha=ha,
                xytext=(x2, yscale * ax_max - (i + 1 + frac) * frac * ax_max),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="0.4",
                    connectionstyle="arc3",
                    lw=1,
                ),
                color=self.spaceblack,
                fontsize=11,
            )
        else:
            i += -1

        LOGGER.debug(
            f"{model}: Text y-pos: {yscale * ax_max - (i + 1 + frac) * frac * ax_max}"
        )
        LOGGER.debug(f"{model}: y-axis max: {ax_max}")

    def _add_cutoff_line(
        self,
        x1,
        ax,
        ax_max,
        ha="right",
        va="bottom",
        outlier_scale=0.99,
        cutoff_scale=1.02,
        yscale=1.0,
    ):
        ax.axvline(x=x1, ymin=0, ymax=yscale, ls="dashed", c=self.spaceblack, alpha=0.7)
        ax.text(
            x1 * outlier_scale,
            yscale * ax_max,
            rf"$\text{{\# of Outliers}} > {x1}$:",
            c=self.spaceblack,
            va=va,
            ha=ha,
            rotation_mode="anchor",
        )

        ax.text(
            x1 * cutoff_scale,
            0.5 * yscale * ax_max,
            "Cutoff",
            rotation=90,
            c=self.spaceblack,
            va="top",
            ha="center",
            rotation_mode="anchor",
        )

    def plot(
        self,
        debug: bool = False,
    ):
        if debug:
            LOGGER.setLevel("DEBUG")

        if self.config.general.mplstyle.get("name") == "radionets":
            import matplotlib as mpl

            mpl.use("pgf")
        elif self.config.general.mplbackend:
            import matplotlib as mpl

            mpl.use(self.config.general.mplbackend)

        with plt.style.context(self.config.general.mplstyle.get("path")):
            # optional overwrite for user settings
            plt.rcParams.update(**self.config.general.rcparams.model_dump())

            for field, val in self.config:
                if val and field not in {"paths", "general", "debug"}:
                    LOGGER.info(f"Plotting {field}:")
                    getattr(self, field)()
