from pathlib import Path

import comet_ml
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.callback.core import Callback, CancelBackwardException
from matplotlib.colors import PowerNorm

from radionets.core.logging import setup_logger
from radionets.core.model import save_model
from radionets.core.utils import _maybe_item, get_ifft_torch
from radionets.evaluation.utils import (
    apply_normalization,
    apply_symmetry,
    check_vmin_vmax,
    eval_model,
    get_ifft,
    get_images,
    load_data,
    load_pretrained_model,
    make_axes_nice,
    rescale_normalization,
)

__all__ = [
    "AvgLossCallback",
    "CometCallback",
    "CudaCallback",
    "DataAug",
    "GradientCallback",
    "Normalize",
    "PredictionImageGradient",
    "SaveTempCallback",
    "SwitchLoss",
]

LOGGER = setup_logger(namespace=__name__)


class CometCallback(Callback):
    """Callback for logging training metrics and visualizations
    to Comet ML.

    This callback logs training and validation losses, and
    creates plots for predictions and Fourier-transformed
    data for monitoring during training.

    Parameters
    ----------
    name : str
        Project name for the Comet ML experiment.
    validation_data : str or Path
        Path to the validation dataset.
    plot_n_epochs : int
        Frequency of plotting (every n epochs).
    amp_phase : bool
        Whether to use amplitude-phase representation.
    scale : str
        Scaling method for data.
    """

    def __init__(self, name, validation_data, plot_n_epochs, amp_phase, scale):
        self.experiment = comet_ml.Experiment(project_name=name)
        self.data_path = validation_data
        self.plot_epoch = plot_n_epochs
        self.test_ds = load_data(self.data_path, mode="test", fourier=True)
        self.amp_phase = amp_phase
        self.scale = scale
        self.uncertainty = False

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
        img_test, img_true, _ = get_images(self.test_ds, 1, rand=False)
        img_test = img_test.unsqueeze(0)
        img_true = img_true.unsqueeze(0)
        model = self.model

        try:
            if self.learn.normalize.mode == "all":
                norm_dict = {"all": 0}
                img_test, norm_dict = apply_normalization(img_test, norm_dict)
        except AttributeError:
            pass

        with self.experiment.test(), torch.no_grad():
            pred = eval_model(img_test, model)

        try:
            if self.learn.normalize.mode == "all":
                pred = rescale_normalization(pred, norm_dict)
        except AttributeError:
            pass

        if pred.shape[1] == 4:
            self.uncertainty = True
            pred = torch.stack((pred[:, 0, :], pred[:, 2, :]), dim=1)

        images = {"pred": pred, "truth": img_true}
        images = apply_symmetry(images)
        pred = images["pred"]
        img_true = images["truth"]

        fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")
        ax = ax.ravel()

        lim_amp = check_vmin_vmax(img_true[0, 0])
        lim_phase = check_vmin_vmax(img_true[0, 1])
        im1 = ax[0].imshow(
            pred[0, 0], cmap="radionets.PuOr", vmin=-lim_amp, vmax=lim_amp
        )
        make_axes_nice(fig, ax[0], im1, "Real")

        im2 = ax[1].imshow(
            pred[0, 1], cmap="radionets.PuOr", vmin=-lim_phase, vmax=lim_phase
        )
        make_axes_nice(fig, ax[1], im2, "Imaginary")

        im3 = ax[2].imshow(
            img_true[0, 0], cmap="radionets.PuOr", vmin=-lim_amp, vmax=lim_amp
        )
        make_axes_nice(fig, ax[2], im3, "Org. Real")

        im4 = ax[3].imshow(
            img_true[0, 1], cmap="radionets.PuOr", vmin=-lim_phase, vmax=lim_phase
        )
        make_axes_nice(fig, ax[3], im4, "Org. Imaginary")

        self.experiment.log_figure(
            figure=fig, figure_name=f"{self.epoch + 1}_pred_epoch"
        )
        plt.close()

    def plot_test_fft(self):
        img_test, img_true, _ = get_images(self.test_ds, 1, rand=False)
        img_test = img_test.unsqueeze(0)
        img_true = img_true.unsqueeze(0)
        model = self.model

        try:
            if self.learn.normalize.mode == "all":
                norm_dict = {"all": 0}
                img_test, norm_dict = apply_normalization(img_test, norm_dict)
        except AttributeError:
            pass

        with self.experiment.test(), torch.no_grad():
            pred = eval_model(img_test, model)

        try:
            if self.learn.normalize.mode == "all":
                pred = rescale_normalization(pred, norm_dict)
        except AttributeError:
            pass

        if self.uncertainty:
            pred = torch.stack((pred[:, 0, :], pred[:, 2, :]), dim=1)

        images = {"pred": pred, "truth": img_true}
        images = apply_symmetry(images)
        pred = images["pred"]
        img_true = images["truth"]

        ifft_pred = get_ifft_torch(
            pred,
            amp_phase=self.amp_phase,
            scale=self.scale,
            uncertainty=self.uncertainty,
        )
        ifft_truth = get_ifft_torch(
            img_true, amp_phase=self.amp_phase, scale=self.scale
        )

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")

        im1 = ax[0].imshow(
            ifft_pred, norm=PowerNorm(0.25, vmax=ifft_truth.max()), cmap="inferno"
        )
        im2 = ax[1].imshow(ifft_truth, norm=PowerNorm(0.25), cmap="inferno")
        a = check_vmin_vmax(ifft_pred - ifft_truth)
        im3 = ax[2].imshow(
            ifft_pred - ifft_truth, cmap="radionets.PuOr", vmin=-a, vmax=a
        )

        make_axes_nice(fig, ax[0], im1, "FFT Prediction")
        make_axes_nice(fig, ax[1], im2, "FFT Truth")
        make_axes_nice(fig, ax[2], im3, "FFT Diff")

        ax[0].set(
            ylabel="Pixels",
            xlabel="Pixels",
        )
        ax[1].set_xlabel("Pixels")
        ax[2].set_xlabel("Pixels")

        self.experiment.log_figure(
            figure=fig, figure_name=f"{self.epoch + 1}_fft_epoch"
        )
        plt.close()

    def after_epoch(self):
        if (self.epoch + 1) % self.plot_epoch == 0:
            self.plot_test_pred()
            self.plot_test_fft()


class AvgLossCallback(Callback):
    """Callback for tracking and plotting average training
    and validation losses.

    Saves the average loss for training and validation
    that is printed to the terminal.
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
        min_epoch = np.argmin(self.loss_valid)
        plt.plot(self.loss_train, label="Training loss")
        plt.plot(self.loss_valid, label="Validation loss")
        plt.axvline(
            min_epoch,
            color="black",
            linestyle="dashed",
            label=f"Minimum at Epoch {min_epoch}",
        )
        plt.xlabel(r"Number of Epochs")
        plt.ylabel(r"Loss")
        plt.legend()

        train = np.array(self.loss_train)
        valid = np.array(self.loss_valid)

        return bool(len(train[train < 0]) == 0 or len(valid[valid < 0]) == 0)

    def plot_lrs(self):
        plt.plot(self.lrs)
        plt.xlabel(r"Number of Batches")
        plt.ylabel(r"Learning rate")


class CudaCallback(Callback):
    """Callback to move model to CUDA device before training.

    Simple callback that ensures the model is moved to the
    GPU before the training loop.

    Attributes
    ----------
    _order : int
        Callback execution order (3).
    """

    _order = 3

    def before_fit(self):
        self.model.cuda()


class DataAug(Callback):
    """Callback that applies data augmentation using
    random rotations.

    Applies random multiples of 90-degree rotations to both
    input and target tensors before each batch to augment
    the training data.
    """

    _order = 3

    def before_batch(self):
        x = self.xb[0].clone()
        y = self.yb[0].clone()

        randint = np.random.randint(0, 1, x.shape[0]) * 2
        last_axis = len(x.shape) - 1

        for i in range(x.shape[0]):
            x[i] = torch.rot90(x[i], int(randint[i]), [last_axis - 2, last_axis - 1])
            y[i] = torch.rot90(y[i], int(randint[i]), [last_axis - 2, last_axis - 1])

        x = x.squeeze(1)
        y = y.squeeze(1)

        self.learn.xb = [x]
        self.learn.yb = [y]


class Normalize(Callback):
    """Normalization callback for input and target data.

    Parameters
    ----------
    conf : dict
        Dictionary containing the normalization type stored
        under the ``'normalize'`` key.
    """

    _order = 4

    def __init__(self, conf):
        self.mode = conf["normalize"]
        if self.mode == "mean":
            self.mean_real = conf["norm_factors"]["mean_real"]
            self.mean_imag = conf["norm_factors"]["mean_imag"]
            self.std_real = conf["norm_factors"]["std_real"]
            self.std_imag = conf["norm_factors"]["std_imag"]

    def normalize(self, x, m, s):
        return (x - m) / s

    def before_batch(self):
        x = self.xb[0].clone()
        y = self.yb[0].clone()

        if self.mode == "max":
            x[:, 0] *= 1 / torch.amax(x[:, 0], dim=(-2, -1), keepdim=True)
            x[:, 1] *= 1 / torch.amax(torch.abs(x[:, 1]), dim=(-2, -1), keepdim=True)
            y[:, 0] *= 1 / torch.amax(x[:, 0], dim=(-2, -1), keepdim=True)
            y[:, 1] *= 1 / torch.amax(torch.abs(x[:, 1]), dim=(-2, -1), keepdim=True)

        elif self.mode == "mean":
            x[:, 0][x[:, 0] != 0] = self.normalize(
                x[:, 0][x[:, 0] != 0], self.mean_real, self.std_real
            )

            x[:, 1][x[:, 1] != 0] = self.normalize(
                x[:, 1][x[:, 1] != 0], self.mean_imag, self.std_imag
            )

            y[:, 0] = self.normalize(y[:, 0], self.mean_real, self.std_real)
            y[:, 1] = self.normalize(y[:, 1], self.mean_imag, self.std_imag)

        elif self.mode == "all":
            # normalize each image so that mean=0 and std=1
            means = x.mean(axis=-1).mean(axis=-1).reshape(x.shape[0], x.shape[1], 1, 1)
            stds = x.std(axis=-1).std(axis=-1).reshape(x.shape[0], x.shape[1], 1, 1)
            x = self.normalize(x, means, stds)
            y = self.normalize(y, means, stds)

        self.learn.xb = [x]
        self.learn.yb = [y]


class SaveTempCallback(Callback):
    """Callback for saving temporary model checkpoints
    during training.

    Parameters
    ----------
    model_path : str or Path
        Path where temporary models will be saved.
    """

    _order = 95

    def __init__(self, model_path):
        self.model_path = model_path

    def after_epoch(self):
        p = Path(self.model_path).parent
        p.mkdir(parents=True, exist_ok=True)

        if (self.epoch + 1) % 10 == 0:
            out = p / f"temp_{self.epoch + 1}.model"
            save_model(self, out)
            LOGGER.info(f"Finished Epoch {self.epoch + 1}, model saved.")


class SwitchLoss(Callback):
    """Callback for switching loss functions during training.

    Changes the loss function to a different one after a specified
    number of epochs.

    Parameters
    ----------
    second_loss : callable
        The loss function to switch to.
    when_switch : int
        Epoch number after which to switch loss functions.
    """

    _order = 5

    def __init__(self, second_loss, when_switch):
        self.second_loss = second_loss
        self.when_switch = when_switch

    def before_epoch(self):
        if (self.epoch + 1) > self.when_switch:
            self.learn.loss_func = self.second_loss


class GradientCallback(Callback):
    """Callback for gradient and prediction tracking.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    validation_data : str or Path
        Path to the validation dataset.
    arch_name : str
        Name of the architecture used for the model.
    amp_phase : bool
        Whether to use amplitude-phase representation.
    """

    def __init__(self, num_epochs, validation_data, arch_name, amp_phase):
        self.num_epochs = num_epochs
        self.data_path = validation_data
        self.test_ds = load_data(
            self.data_path, mode="test", fourier=True, source_list=False
        )
        self.arch_name = arch_name
        self.amp_phase = amp_phase

    def before_backward(self):
        raise CancelBackwardException

    def after_cancel_backward(self):
        self.learn.loss.backward()

        # access gradients of weights of layers (with specified batch and epoch)
        if self.epoch == self.num_epochs - 1 and self.iter == self.n_iter - 1:
            grads = []
            for param in self.learn.model.parameters():
                grads.append(param.grad.view(-1))
        # print or save

    def after_epoch(self):
        img_test, img_true = get_images(self.test_ds, 1, rand=False)

        # for each epoch put test image through model and save to csv
        fname_template = "pred_{i}.csv"
        np.savetxt(
            fname_template.format(i=self.epoch),
            get_ifft(eval_model(img_test, self.model), self.amp_phase),
            delimiter=",",
        )

        # # fourier space
        amp_names = "pred_amp_{i}.csv"
        phase_names = "pred_phase_{i}.csv"
        output = eval_model(img_test, self.model)
        np.savetxt(
            amp_names.format(i=self.epoch), output[0][0].cpu().numpy(), delimiter=","
        )
        np.savetxt(
            phase_names.format(i=self.epoch), output[0][1].cpu().numpy(), delimiter=","
        )


class PredictionImageGradient(Callback):
    """Callback for computing spatial gradients
    of model predictions.

    Parameters
    ----------
    validation_data : str or Path
        Path to validation dataset.
    model : str or Path
        Path to pretrained model.
    amp_phase : bool
        Whether to use amplitude-phase representation.
    arch_name : str
        Name of the architecture used for the model.
    """

    def __init__(self, validation_data, model, amp_phase, arch_name):
        self.data_path = validation_data
        self.test_ds = load_data(
            self.data_path, mode="test", fourier=True, source_list=False
        )
        self.model = model
        self.amp_phase = amp_phase
        self.arch_name = arch_name

    def save_output_pred(self):
        img_test, img_true = get_images(self.test_ds, 5, rand=False)

        img_size = img_test[0].shape[-1]
        model_used = load_pretrained_model(self.arch_name, self.model, img_size)

        output = eval_model(img_test[0], model_used)
        gradient = K.filters.spatial_gradient(output)

        grads_x = get_ifft(gradient[:, :, 0], self.amp_phase)
        grads_y = get_ifft(gradient[:, :, 1], self.amp_phase)

        return grads_x, grads_y
