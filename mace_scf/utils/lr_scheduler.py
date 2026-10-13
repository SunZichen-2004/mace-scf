"""Learning-rate schedulers used by mace-scf training."""

import logging

import torch


class CosineAnnealingScheduler:
    """Cosine anneal from the optimizer's current learning rate to ``eta_min``.

    ``step`` accepts ``metrics`` and ``epoch`` so it can be called the same way
    as the plateau scheduler. The cosine length is one training stage.
    """

    def __init__(self, optimizer, t_max: int, eta_min: float) -> None:
        if t_max < 1:
            raise ValueError(f"CosineAnnealingLR T_max must be >= 1, got {t_max}")
        self.scheduler = "CosineAnnealingLR"
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max),
            eta_min=float(eta_min),
        )
        logging.info(
            "CosineAnnealingLR: T_max=%s, eta_min=%s, base_lrs=%s",
            t_max,
            eta_min,
            [group["lr"] for group in optimizer.param_groups],
        )

    def step(self, metrics=None, epoch=None):
        del metrics, epoch
        self.lr_scheduler.step()
        logging.info(
            "CosineAnnealingLR: lr = %.6e", self.lr_scheduler.get_last_lr()[0]
        )

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)
