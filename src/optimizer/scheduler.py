import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
    """

    def __init__(self, optimizer, warmup_epoch, after_scheduler):
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False

        if isinstance(self.after_scheduler, ReduceLROnPlateau):
            raise NotImplementedError
        assert self.after_scheduler is not None

        super().__init__(optimizer)

    def get_lr(self):
        assert not self.finished

        if self.last_epoch >= self.warmup_epoch:
            self.after_scheduler.base_lrs = self.base_lrs
            self.finished = True
            return self.after_scheduler.get_last_lr()

        return [
            base_lr * (float(self.last_epoch + 1) / self.warmup_epoch) for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)

    def state_dict(self):
        state = {}
        for key, value in self.__dict__.items():
            if key == "optimizer":
                continue
            elif key == "after_scheduler":
                state[key] = self.after_scheduler.state_dict()
            else:
                state[key] = value
        return state

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict["after_scheduler"])
        del state_dict["after_scheduler"]
        super().load_state_dict(state_dict)


class PolynomialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps, end_lr=0.0001, power=1.0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_lr)
            * ((1 - min(self.last_epoch, self.decay_steps) / self.decay_steps) ** self.power)
            + self.end_lr
            for base_lr in self.base_lrs
        ]


def wrap_warmup(scheduler, times, **kwargs):
    return GradualWarmupScheduler(
        scheduler.optimizer, warmup_epoch=times, after_scheduler=scheduler
    )


def create_scheduler(optimizer, sch_times=None, name=None, policy="epoch", warmup=None, **kwargs):
    if name == "step":
        sch = optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif name == "cosine":
        sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_times, **kwargs)
    elif name == "poly":
        sch = PolynomialDecay(optimizer, sch_times, **kwargs)
    else:
        raise NotImplementedError

    if warmup:
        sch = wrap_warmup(sch, **warmup)
    if policy == "iteration":
        sch = dict(scheduler=sch, interval="step")

    return sch
