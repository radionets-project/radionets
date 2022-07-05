import torch
import numpy as np
import pandas as pd
from radionets.dl_framework.data import do_normalisation
from radionets.dl_framework.logger import make_notifier
from radionets.dl_framework.model import save_model
from radionets.dl_framework.utils import _maybe_item
from fastai.callback.core import Callback
from pathlib import Path
from fastcore.foundation import L
import matplotlib.pyplot as plt
from radionets.evaluation.utils import (
    load_data,
    get_images,
    eval_model,
    make_axes_nice,
    check_vmin_vmax,
)
from radionets.evaluation.plotting import create_OrBu
from radionets.dl_framework.utils import get_ifft_torch

OrBu = create_OrBu()


class CometCallback(Callback):
    def __init__(self, name, test_data, plot_n_epochs, amp_phase, scale):
        from comet_ml import Experiment

        self.experiment = Experiment(project_name=name)
        self.data_path = test_data
        self.plot_epoch = plot_n_epochs
        self.test_ds = load_data(
            self.data_path,
            mode="test",
            fourier=True,
            source_list=False,
        )
        self.amp_phase = amp_phase
        self.scale = scale

    def after_train(self):
        self.experiment.log_metric(
            "Train Loss",
            self.recorder._train_mets.map(_maybe_item),
            epoch=self.epoch + 1,
        )

    def after_validate(self):
        self.experiment.log_metric(
            "Validation Loss",
            self.recorder._valid_mets.map(_maybe_item),
            epoch=self.epoch + 1,
        )

    def plot_test_pred(self):
        img_test, img_true = get_images(self.test_ds, 1, norm_path="none", rand=False)
        model = self.model
        with self.experiment.test():
            with torch.no_grad():
                pred = eval_model(img_test, model)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        lim_phase = check_vmin_vmax(img_true[0, 1])
        im1 = ax1.imshow(pred[0, 0], cmap="inferno")
        im2 = ax2.imshow(pred[0, 1], cmap=OrBu, vmin=-lim_phase, vmax=lim_phase)
        im3 = ax3.imshow(img_true[0, 0], cmap="inferno")
        im4 = ax4.imshow(img_true[0, 1], cmap=OrBu, vmin=-lim_phase, vmax=lim_phase)
        make_axes_nice(fig, ax1, im1, "Amplitude")
        make_axes_nice(fig, ax2, im2, "Phase", phase=True)
        make_axes_nice(fig, ax3, im3, "Org. Amplitude")
        make_axes_nice(fig, ax4, im4, "Org. Phase", phase=True)
        fig.tight_layout(pad=0.1)
        self.experiment.log_figure(
            figure=fig, figure_name=f"{self.epoch + 1}_pred_epoch"
        )
        plt.close("all")

    def plot_test_fft(self):
        img_test, img_true = get_images(self.test_ds, 1, norm_path="none", rand=False)
        model = self.model
        with self.experiment.test():
            with torch.no_grad():
                pred = eval_model(img_test, model)

        ifft_pred = get_ifft_torch(pred, amp_phase=self.amp_phase, scale=self.scale)
        ifft_truth = get_ifft_torch(
            img_true, amp_phase=self.amp_phase, scale=self.scale
        )

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10))
        im1 = ax1.imshow(ifft_pred, vmax=ifft_truth.max(), cmap="inferno")
        im2 = ax2.imshow(ifft_truth, cmap="inferno")
        a = check_vmin_vmax(ifft_pred - ifft_truth)
        im3 = ax3.imshow(ifft_pred - ifft_truth, cmap=OrBu, vmin=-a, vmax=a)

        make_axes_nice(fig, ax1, im1, r"FFT Prediction")
        make_axes_nice(fig, ax2, im2, r"FFT Truth")
        make_axes_nice(fig, ax3, im3, r"FFT Diff")

        ax1.set_ylabel(r"Pixels")
        ax1.set_xlabel(r"Pixels")
        ax2.set_xlabel(r"Pixels")
        ax3.set_xlabel(r"Pixels")

        fig.tight_layout(pad=0.1)
        self.experiment.log_figure(
            figure=fig, figure_name=f"{self.epoch + 1}_fft_epoch"
        )
        plt.close("all")

    def after_epoch(self):
        if (self.epoch + 1) % self.plot_epoch == 0:
            self.plot_test_pred()
            self.plot_test_fft()


class TelegramLoggerCallback(Callback):
    def __init__(self, model_name):
        self.model_name = model_name

    def before_fit(self):
        tlogger = make_notifier()
        tlogger.info(f"Start des Trainings von Modell {self.model_name}")

    def after_epoch(self):
        if (self.epoch + 1) % 10 == 0:
            tlogger = make_notifier()
            tlogger.info(
                "{}: Epoche {}/{} mit Loss {}".format(
                    self.model_name,
                    self.epoch + 1,
                    self.n_epoch,
                    L(self.recorder.values[0:]).itemgot(1)[-1],
                )
            )

    def after_fit(self):
        tlogger = make_notifier()
        tlogger.info(
            "{}: Ende des Trainings nach {} Epochen mit Loss {}".format(
                self.model_name,
                self.epoch + 1,
                L(self.recorder.values[0:]).itemgot(1)[-1],
            )
        )


class AvgLossCallback(Callback):
    """Save the same average Loss for training and validation as printed to
    the terminal.

    Parameters
    ----------
    Callback : object
        Callback class
    """

    def __init__(self):
        if not hasattr(self, "loss_train"):
            self.loss_train = []
        if not hasattr(self, "loss_valid"):
            self.loss_valid = []
        if not hasattr(self, "lrs"):
            self.lrs = []

    def after_train(self):
        self.loss_train.append(self.recorder._train_mets.map(_maybe_item))

    def after_validate(self):
        self.loss_valid.append(self.recorder._valid_mets.map(_maybe_item))

    def after_batch(self):
        self.lrs.append(self.opt.hypers[-1]["lr"])

    def plot_loss(self):
        plt.plot(self.loss_train, label="Training loss")
        plt.plot(self.loss_valid, label="Validation loss")
        plt.xlabel(r"Number of Epochs")
        plt.ylabel(r"Loss")
        plt.legend()
        plt.tight_layout()

        train = np.array(self.loss_train)
        valid = np.array(self.loss_valid)
        if len(train[train < 0]) == 0 or len(valid[valid < 0]) == 0:
            return True
        else:
            return False

    def plot_lrs(self):
        plt.plot(self.lrs)
        plt.xlabel(r"Number of Batches")
        plt.ylabel(r"Learning rate")
        plt.tight_layout()


class NormCallback(Callback):
    _order = 2

    def __init__(self, norm_path):
        self.path = norm_path

    def before_batch(self):
        self.learn.xb = [self.normalize_tfm()]

    def normalize_tfm(self):
        norm = pd.read_csv(self.path)
        a = do_normalisation(self.learn.xb[0].clone(), norm)
        assert self.learn.xb[0][:, 0].mean() != a[:, 0].mean()
        # mean for imag and phase is approx 0
        # assert x[:, 1].mean() != a[:, 1].mean()
        return a


class CudaCallback(Callback):
    _order = 3

    def before_fit(self):
        self.model.cuda()


class DataAug(Callback):
    _order = 3

    def before_batch(self):
        x = self.xb[0].clone()
        y = self.yb[0].clone()
        randint = np.random.randint(0, 4, x.shape[0])
        last_axis = len(x.shape) - 1
        for i in range(x.shape[0]):
            x[i] = torch.rot90(x[i], int(randint[i]), [last_axis - 2, last_axis - 1])
            y[i] = torch.rot90(y[i], int(randint[i]), [last_axis - 2, last_axis - 1])
        x = x.squeeze(1)
        y = y.squeeze(1)
        self.learn.xb = [x]
        self.learn.yb = [y]


class SaveTempCallback(Callback):
    _order = 95

    def __init__(self, model_path):
        self.model_path = model_path

    def after_epoch(self):
        p = Path(self.model_path).parent
        p.mkdir(parents=True, exist_ok=True)
        if (self.epoch + 1) % 10 == 0:
            out = p / f"temp_{self.epoch + 1}.model"
            save_model(self, out)
            print(f"\nFinished Epoch {self.epoch + 1}, model saved.\n")


class SwitchLoss(Callback):
    _order = 5

    def __init__(self, second_loss, when_switch):
        self.second_loss = second_loss
        self.when_switch = when_switch

    def before_epoch(self):
        if (self.epoch + 1) > self.when_switch:
            self.learn.loss_func = self.second_loss
