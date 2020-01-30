from dl_framework.utils import listify, compose
from functools import partial
import torch
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from torch._six import container_abcs


def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p


def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr*wd)
    return p


weight_decay._defaults = dict(wd=0.)


def l2_reg(p, lr, wd, **kwargs):
    p.grad.data.add_(wd, p.data)
    return p


l2_reg._defaults = dict(wd=0.)


def maybe_update(os, dest, f):
    for o in os:
        for k, v in f(o).items():
            if k not in dest:
                dest[k] = v


def get_defaults(d):
    return getattr(d, '_defaults', {})


class Optimizer():
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [(p, hyper) for pg, hyper in zip(self.param_groups, self.hypers)
                for p in pg if p.grad is not None]

    def zero_grad(self):
        for p, hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p, hyper in self.grad_params():
            compose(p, self.steppers, **hyper)

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})


sgd_opt = partial(Optimizer, steppers=[weight_decay, sgd_step])


class StatefulOptimizer(Optimizer):
    def __init__(self, params, steppers, stats=None, **defaults):
        self.stats = listify(stats)
        maybe_update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {}

    def step(self):
        for p, hyper in self.grad_params():
            if p not in self.state:
                # Create a state for p and call all the statistics to
                # initialize it.
                self.state[p] = {}
                maybe_update(self.stats, self.state[p],
                             lambda o: o.init_state(p))
            state = self.state[p]
            for stat in self.stats:
                state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state


class Stat():
    _defaults = {}
    def init_state(self, p): raise NotImplementedError
    def update(self, p, state, **kwargs): raise NotImplementedError


def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)
    return p


class AverageGrad(Stat):
    _defaults = dict(mom=0.9)

    def __init__(self, dampening: bool = False):
        self.dampening = dampening

    def init_state(self, p):
        return {'grad_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, mom, **kwargs):
        state['mom_damp'] = 1-mom if self.dampening else 1.
        state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
        return state


class AverageSqrGrad(Stat):
    _defaults = dict(sqr_mom=0.99)

    def __init__(self, dampening: bool = True):
        self.dampening = dampening

    def init_state(self, p):
        return {'sqr_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, sqr_mom, **kwargs):
        state['sqr_damp'] = 1-sqr_mom if self.dampening else 1.
        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data,
                                                p.grad.data)
        return state


class StepCount(Stat):
    def init_state(self, p):
        return {'step': 0}

    def update(self, p, state, **kwargs):
        state['step'] += 1
        return state


def debias(mom, damp, step):
    return damp * (1 - mom**step) / (1-mom)


def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp,
              grad_avg, sqr_avg, eps, **kwargs):
    debias1 = debias(mom,     mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
    return p


adam_step._defaults = dict(eps=1e-5)


def adam_opt(xtra_step=None, **kwargs):
    return partial(StatefulOptimizer, steppers=[adam_step,
                   weight_decay] + listify(xtra_step),
                   stats=[AverageGrad(dampening=True),
                   AverageSqrGrad(), StepCount()], **kwargs)
