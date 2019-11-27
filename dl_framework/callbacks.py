from re import sub
from dl_framework.utils import camel2snake, AvgStats, listify, lin_comb
import torch
import matplotlib.pyplot as plt
from functools import partial
from torch.distributions.beta import Beta
import pandas as pd
from dl_framework.data import do_normalisation


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Callback():
    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1./self.iters
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
        self.logger(self.train_stats)
        self.logger(self.valid_stats)


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = listify(sched_funcs)

    def begin_batch(self):
        if not self.in_train:
            return
        fs = self.sched_funcs
        if len(fs) == 1:
            fs = fs*len(self.opt.param_groups)
        pos = self.n_epochs/self.epochs
        for f, h in zip(fs, self.opt.hypers):
            h[self.pname] = f(pos)


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.hypers[-1]['lr'])

    def after_epoch(self):
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_loss(self):
        plt.plot(self.losses)

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.hypers:
            pg['lr'] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss*10:
            return CancelTrainException
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()

    def begin_batch(self):
        self.run.xb = self.run.xb.cuda()
        self.run.yb = self.run.yb.cuda()


class BatchTransformXCallback(Callback):
    _order = 2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb = self.tfm(self.run.xb)


def view_tfm(*size):
    def _inner(x):
        """
        add correct shape (bs, #channels, shape of array)
        """
        a = x.view(*((-1,)+size))
        return a
    return _inner


def normalize_tfm(norm_path):
    def _inner(x):
        norm = pd.read_csv(norm_path)
        x = do_normalisation(x, norm)
        return x
    return _inner


# mix-up

class NoneReduce():
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.old_red = None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else:
            return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None:
            setattr(self.loss_func, 'reduction', self.old_red)


def unsqueeze(input, dims):
    for dim in listify(dims):
        input = torch.unsqueeze(input, dim)
    return input


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() \
        if reduction == 'sum' else loss


class MixUp(Callback):
    _order = 90  # Runs after normalization and cuda

    def __init__(self, α: float = 0.4):
        self.distrib = Beta(torch.tensor([α],
                            dtype=torch.float),
                            torch.tensor([α], dtype=torch.float))

    def begin_fit(self):
        self.old_loss_func = self.run.loss_func
        self.run.loss_func = self.loss_func

    def begin_batch(self):
        if not self.in_train:
            return  # Only mixup things during training
        λ = self.distrib.sample((self.yb.size(0),)
                                ).squeeze().to(self.xb.device)
        λ = torch.stack([λ, 1-λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], [1, 1, 1])
        self.λ2 = unsqueeze(λ.max(1)[0], [1])
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1, self.yb1 = self.xb[shuffle], self.yb[shuffle]
        self.run.xb = lin_comb(self.xb, xb1, self.λ)
        # self.run.yb = lin_comb(self.yb, self.yb1, self.λ2)
        # img = self.run.xb[0].squeeze(0).cpu()
        # plt.imshow(img, cmap='RdBu', vmin=-img.max(), vmax=img.max())

    def after_fit(self): self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        loss = lin_comb(loss1, loss2, self.λ)
        return reduce_loss(loss,
                           getattr(self.old_loss_func, 'reduction', 'mean'))


class SaveCallback(Callback):
    _order = 95

    def after_epoch(self):
        if round(self.n_epochs) % 10 == 0:
            state = self.model.state_dict()
            torch.save(state, './models/temp.model')
            print('\nFinished Epoch {}, model saved.\n'.format(round(self.n_epochs)))
