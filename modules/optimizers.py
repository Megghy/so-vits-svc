import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """EvoLved Sign Momentum (Chen et al. 2023). lr 取 AdamW 的 1/3~1/10,配合更大 weight_decay。
    支持多 param_groups(不同 lr),params 可传 iterable 或 list of dicts。"""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad, state = p.grad, self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def build_optimizer(params, t, vocoder_lr=None):
    """按 config.train.optimizer 选优化器;缺省 adamw,保持 torch 默认 weight_decay。
    vocoder_lr: 若非 None,params 必须是 [main_params, vocoder_params] 列表,构建双 lr 组。"""
    name = (t.get("optimizer") or "adamw").lower()
    wd = t.get("weight_decay")

    if vocoder_lr is not None:
        if not isinstance(params, list) or len(params) != 2:
            raise ValueError("vocoder_lr 非空时 params 必须是 [main_params, vocoder_params]")
        param_groups = [
            {"params": params[0], "lr": t.learning_rate},
            {"params": params[1], "lr": vocoder_lr}
        ]
        if name == "lion":
            return Lion(param_groups, lr=t.learning_rate, betas=tuple(t.betas),
                        weight_decay=wd if wd is not None else 0.0)
        kwargs = {} if wd is None else {"weight_decay": wd}
        return torch.optim.AdamW(param_groups, lr=t.learning_rate, betas=tuple(t.betas), eps=t.eps, **kwargs)

    if name == "lion":
        return Lion(params, t.learning_rate, betas=tuple(t.betas),
                    weight_decay=wd if wd is not None else 0.0)
    kwargs = {} if wd is None else {"weight_decay": wd}
    return torch.optim.AdamW(params, t.learning_rate, betas=t.betas, eps=t.eps, **kwargs)
