import re
from typing import Iterable
from torch import nn
from dl_framework.hooks import Hooks


def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def listify(o):
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self):
        """
        Mit item wird nur der der tatsächliche Wert abgegriffen,
        statt tensor(3561161, device='cuda:0').
        self.tot_mets ist eine Liste von der gleichen Struktur wie oben,
        und mit dem '+' wird die zweite Liste an das Ende der ersten Liste
        angehängt
        """
        print("\ntot_loss: ", self.tot_loss)
        print("tot_mets: ", self.tot_mets[0])
        print("addition: ", [self.tot_loss.item()] + [self.tot_mets[0].item()], "\n")
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        """
        Hier wird durch die totale Anzahl der Bilder geteilt, die
        bisher verwendet wurde, um einen Ausdruck für den Mittelwert
        zu erhalten.
        """
        return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f'{"train" if self.in_train else "valid"}: {self.avg_stats}'

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


def param_getter(m):
    return m.parameters()


def get_batch(dl, learn):
    learn.xb, learn.yb = next(iter(dl))
    for cb in learn.cbs:
        cb.set_runner(learn)
    learn('begin_batch')
    return learn.xb, learn.yb


def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)


def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


def model_summary(run, learn, data, find_all=False):
    xb, yb = get_batch(data.valid_dl, run)
    # Model may not be on the GPU yet
    device = next(learn.model.parameters()).device
    xb, yb = xb.to(device), yb.to(device)
    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    f = lambda hook, mod, inp, out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks:
        learn.model(xb)


class ListContainer():
    def __init__(self, items): self.items = listify(items)

    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)    # bool mask
                return [o for m, o in zip(idx, self.items) if m]
            return [self.items[i] for i in idx]

    def __len__(self): return len(self.items)

    def __iter__(self): return iter(self.items)

    def __setitem__(self, i, o): self.items[i] = o

    def __delitem__(self, i): del(self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10:
            res = res[:-1] + '...]'
        return res
