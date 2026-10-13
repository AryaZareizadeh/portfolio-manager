"""Learning-rate policies shared by recurrent, Kronos and tokenizer training."""

import math

import torch

POLICIES = ("constant", "warmup_cosine", "plateau", "step", "linear", "onecycle")
FIELDS = ("lr_scheduler", "lr_min_ratio", "lr_warmup_fraction", "lr_factor", "lr_step_epochs", "lr_patience")


def validate_schedule(config, prefix=""):
    def value(name):
        return getattr(config, prefix + name)
    if value("lr_scheduler") not in POLICIES:
        raise ValueError(f"{prefix}lr_scheduler must be one of {', '.join(POLICIES)}.")
    for name, low, high, inclusive_low in (("lr_min_ratio", 0, 1, True),
                                         ("lr_warmup_fraction", 0, 1, True),
                                         ("lr_factor", 0, 1, False)):
        number = value(name)
        if (type(number) not in (int, float) or not math.isfinite(number)
                or not (low <= number < high if inclusive_low else low < number < high)):
            raise ValueError(f"{prefix}{name} must be finite and {'at least' if inclusive_low else 'greater than'} {low}, and below {high}.")
    for name, minimum in (("lr_step_epochs", 1), ("lr_patience", 0)):
        if type(value(name)) is not int or value(name) < minimum:
            raise ValueError(f"{prefix}{name} must be an integer of at least {minimum}.")


class LearningRateSchedule:
    """Smooth policies advance after each update; adaptive policies after validation.

    The last planned update reaches the minimum. Continuation holds that rate
    rather than restarting a cycle. OneCycle changes learning rate only, keeping
    the project's existing AdamW momentum settings.
    """

    def __init__(self, optimizer, config, epochs, steps_per_epoch, prefix=""):
        validate_schedule(config, prefix)
        self.optimizer = optimizer
        self.policy = getattr(config, prefix + "lr_scheduler")
        self.peak = optimizer.param_groups[0]["lr"]
        self.minimum = self.peak * getattr(config, prefix + "lr_min_ratio")
        self.total = max(1, epochs * steps_per_epoch)
        self.warmup = min(max(0, math.ceil(self.total * getattr(config, prefix + "lr_warmup_fraction"))),
                          max(0, self.total - 2))
        self.factor = getattr(config, prefix + "lr_factor")
        self.step_epochs = getattr(config, prefix + "lr_step_epochs")
        self.updates = self.epochs = 0
        self.last_used = self.peak
        self.plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.factor, patience=getattr(config, prefix + "lr_patience"),
            min_lr=self.minimum) if self.policy == "plateau" else None)
        if self.policy in ("warmup_cosine", "linear", "onecycle"):
            self._set(self._planned_rate(0))

    @property
    def rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def _set(self, rate):
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def _planned_rate(self, update):
        if self.total == 1:
            return self.peak if update == 0 else self.minimum
        if update >= self.total - 1:
            return self.minimum
        if self.policy != "linear" and self.warmup and update < self.warmup:
            fraction = update / self.warmup
            if self.policy == "onecycle":
                fraction = (1 - math.cos(math.pi * fraction)) / 2
            start_rate = max(self.minimum, self.peak * 0.1)
            return start_rate + (self.peak - start_rate) * fraction
        start = 0 if self.policy == "linear" else self.warmup
        fraction = (update - start) / (self.total - 1 - start)
        scale = 1 - fraction if self.policy == "linear" else (1 + math.cos(math.pi * fraction)) / 2
        return self.minimum + (self.peak - self.minimum) * scale

    def batch_finished(self):
        self.last_used = self.rate
        self.updates += 1
        if self.policy in ("warmup_cosine", "linear", "onecycle"):
            self._set(self._planned_rate(self.updates))

    def epoch_finished(self, validation_loss):
        self.epochs += 1
        if self.plateau is not None:
            self.plateau.step(validation_loss)
        elif self.policy == "step" and self.epochs % self.step_epochs == 0:
            self._set(max(self.minimum, self.rate * self.factor))
