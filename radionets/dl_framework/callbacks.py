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
    """Save the same average Loss for training and validation as printed to the terminal.

    Parameters
    ----------
    Callback : object
        Callback class
    """

    def __init__(self):
        self.loss_train = []
        self.loss_valid = []

    def after_train(self):
        self.loss_train.append(self.recorder._train_mets.map(_maybe_item))

    def after_validate(self):
        self.loss_valid.append(self.recorder._valid_mets.map(_maybe_item))

    def plot_loss(self):
        plt.plot(self.loss_train, label="train")
        plt.plot(self.loss_valid, label="valid")
        plt.xlabel(r"Number of Epochs")
        plt.ylabel(r"Loss")
        plt.legend()
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


# class BatchTransformXCallback(Callback):
#     _order = 2

#     def __init__(self, tfm):
#         self.tfm = tfm

#     def batch_begin(self):
#         print("TESTBATCH")
#         self.run.xb = self.normalize_tfm(self)

#     def normalize_tfm(self):
#         print(self.path)
#         norm = pd.read_csv(self.path)
#         a = do_normalisation(self.run.xb.clone(), norm)
#         assert self.run.xb[:, 0].mean() != a[:, 0].mean()
#         # mean for imag and phase is approx 0
#         # assert x[:, 1].mean() != a[:, 1].mean()
#         return a


# def view_tfm(*size):
#     def _inner(x):
#         """
#         add correct shape (bs, #channels, shape of array)
#         """
#         a = x.view(*((-1,) + size))
#         return a

#     return _inner


# def normalize_tfm(self):
#     print(norm_path)
#     norm = pd.read_csv(norm_path)
#     a = do_normalisation(x.clone(), norm)
#     assert x[:, 0].mean() != a[:, 0].mean()
#     # mean for imag and phase is approx 0
#     # assert x[:, 1].mean() != a[:, 1].mean()
#     return a


def zero_imag():
    def _inner(x):
        a = x
        imag = a[:, 1, :]
        num = 0
        for i in range(imag.shape[0]):
            if imag[i].max() < 1e-9:
                # print(imag[i].mean().item())
                num += 1
                imag[i] = torch.zeros(imag.shape[1])
        a[:, 1, :] = imag
        # print(num)
        return a

    return _inner


class DataAug(Callback):
    _order = 3

    def begin_batch(self):
        x = self.run.xb.clone()
        y = self.run.yb.clone()
        randint = np.random.randint(0, 4, x.shape[0])
        for i in range(x.shape[0]):
            x[i, 0] = torch.rot90(x[i, 0], int(randint[i]))
            x[i, 1] = torch.rot90(x[i, 1], int(randint[i]))
            y[i, 0] = torch.rot90(y[i, 0], int(randint[i]))
            y[i, 1] = torch.rot90(y[i, 1], int(randint[i]))
        self.run.xb = x
        self.run.yb = y


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
