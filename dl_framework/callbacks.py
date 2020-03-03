from re import sub
from dl_framework.utils import camel2snake, AvgStats, listify, lin_comb
import torch
import matplotlib.pyplot as plt
from functools import partial
from torch.distributions.beta import Beta
import pandas as pd
from dl_framework.data import do_normalisation
from dl_framework.logger import make_notifier
from dl_framework.model import save_model


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Callback:
    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class TrainEvalCallback(Callback):
    _order = 0

    def begin_fit(self):
        self.run.n_epochs = 0.0
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        # We use the logger function of the `Learner` here, it can be
        # customized to write in a file or in a progress bar
        self.log(self.train_stats.avg_print())
        self.log(self.valid_stats.avg_print())


class ParamScheduler(Callback):
    _order = 3

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = listify(sched_funcs)

    def begin_batch(self):
        if not self.in_train:
            return
        fs = self.sched_funcs
        if len(fs) == 1:
            fs = fs * len(self.opt.param_groups)
        pos = self.n_epochs / self.epochs
        for f, h in zip(fs, self.opt.hypers):
            h[self.pname] = f(pos)


class Recorder(Callback):
    _order = 5

    def begin_fit(self):
        if not hasattr(self, "lrs"):
            self.lrs = []
        if not hasattr(self, "train_losses"):
            self.train_losses = []
        if not hasattr(self, "valid_losses"):
            self.valid_losses = []
        if not hasattr(self, "losses"):
            self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.state_dict()["param_groups"][0]["lr"])

    def after_epoch(self):
        self.train_losses.append(self.avg_stats.train_stats.avg_stats[1])
        self.valid_losses.append(self.avg_stats.valid_stats.avg_stats[1])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_loss(self):
        plt.plot(self.train_losses, label="train loss")
        plt.plot(self.valid_losses, label="valid loss")
        # plt.plot(self.losses, label="loss")
        plt.yscale("log")
        plt.xlabel(r"Number of Epochs")
        plt.ylabel(r"Loss")
        plt.legend()
        plt.tight_layout()

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.xlabel(r"learning rate")
        plt.ylabel(r"loss")
        plt.plot(self.lrs[:n], losses[:n])
        plt.show()


class Recorder_lr_find(Callback):
    """
    Recorder class for the lr_find. Main difference between the recorder
    and this class is that the loss is appended after each batch and not
    after each epoch.
    """

    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.state_dict()["param_groups"][0]["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot(self, skip_last=0, save=False):
        losses = [o.item() for o in self.losses]
        n = len(losses) - skip_last
        plt.plot(self.lrs[:n], losses[:n])
        plt.xscale("log")
        plt.xlabel(r"learning rate")
        plt.ylabel(r"loss")
        if save is False:
            plt.show()


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        """
        max_iter should be slightly bigger than the number of batches.
        Only this way maximum and minimum learning rate are set.
        """
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.run.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            return CancelTrainException
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class LoggerCallback(Callback):
    def __init__(self, model_name):
        self.model_name = model_name

    def begin_fit(self):
        logger = make_notifier()
        logger.info("Start des Trainings von Modell {}".format(self.model_name))

    def after_epoch(self):
        if (self.epoch + 1) % 10 == 0:
            logger = make_notifier()
            logger.info(
                "{}: Epoche {}/{} mit Loss {}".format(
                    self.model_name,
                    self.epoch + 1,
                    self.epochs,
                    self.avg_stats.valid_stats.avg_stats[1],
                )
            )

    def after_fit(self):
        logger = make_notifier()
        logger.info(
            "{}: Ende des Trainings nach {} Epochen mit Loss {}".format(
                self.model_name, self.epoch + 1, self.avg_stats.valid_stats.avg_stats[1]
            )
        )


class CudaCallback(Callback):
    def begin_fit(self):
        self.model.cuda()

    def begin_batch(self):
        self.run.xb = self.run.xb.cuda()
        self.run.yb = self.run.yb.cuda()


class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.run.xb)


def view_tfm(*size):
    def _inner(x):
        """
        add correct shape (bs, #channels, shape of array)
        """
        a = x.view(*((-1,) + size))
        return a

    return _inner


def normalize_tfm(norm_path):
    def _inner(x):
        norm = pd.read_csv(norm_path)
        x = do_normalisation(x, norm)
        return x

    return _inner


def zero_imag():
    def _inner(x):
        a = x
        imag = a[:, 1, :]
        num = 0
        for i in range(imag.shape[0]):
            if imag[i].mean().abs() < 1e-5:
                # print(imag[i].mean().item())
                num += 1
                imag[i] = torch.zeros(imag.shape[1])
        a[:, 1, :] = imag
        # print(num)
        return a

    return _inner


# mix-up


class NoneReduce:
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.old_red = None

    def __enter__(self):
        if hasattr(self.loss_func, "reduction"):
            self.old_red = getattr(self.loss_func, "reduction")
            setattr(self.loss_func, "reduction", "none")
            return self.loss_func
        else:
            return partial(self.loss_func, reduction="none")

    def __exit__(self, type, value, traceback):
        if self.old_red is not None:
            setattr(self.loss_func, "reduction", self.old_red)


def unsqueeze(input, dims):
    for dim in listify(dims):
        input = torch.unsqueeze(input, dim)
    return input


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class MixUp(Callback):
    _order = 90  # Runs after normalization and cuda

    def __init__(self, α: float = 0.4):
        self.distrib = Beta(
            torch.tensor([α], dtype=torch.float), torch.tensor([α], dtype=torch.float)
        )

    def begin_fit(self):
        self.old_loss_func = self.run.loss_func
        self.run.loss_func = self.loss_func

    def begin_batch(self):
        if not self.in_train:
            return  # Only mixup things during training
        λ = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        λ = torch.stack([λ, 1 - λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], [1, 1, 1])
        self.λ2 = unsqueeze(λ.max(1)[0], [1])
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1, self.yb1 = self.xb[shuffle], self.yb[shuffle]
        self.run.xb = lin_comb(self.xb, xb1, self.λ)
        # self.run.yb = lin_comb(self.yb, self.yb1, self.λ2)
        # img = self.run.xb[0].squeeze(0).cpu()
        # plt.imshow(img, cmap='RdBu', vmin=-img.max(), vmax=img.max())

    def after_fit(self):
        self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        loss = lin_comb(loss1, loss2, self.λ)
        return reduce_loss(loss, getattr(self.old_loss_func, "reduction", "mean"))


class SaveCallback(Callback):
    _order = 95

    def __init__(self, model_path):
        self.model_path = "/".join(model_path.split("/", 2)[:2])

    def after_epoch(self):
        if round(self.n_epochs) % 10 == 0:
            save_model(
                self, self.model_path + "/temp_{}.model".format(round(self.n_epochs))
            )
            print("\nFinished Epoch {}, model saved.\n".format(round(self.n_epochs)))
