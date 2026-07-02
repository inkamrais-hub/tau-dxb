"""
τ* solver — standalone, reusable operator for finding the optimal
temperature τ of the τ-opus attention normalization.

The estimator solves, per attention head:

    τ* = argmin_τ  KL( softmax(scores) || softmax(tau * log_sigma(scores)) )

where sigma is a positive transfer function (softplus / sigmoid / exp / relu).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F


SigmaType = Literal["softplus", "sigmoid", "exp", "relu"]


def _log_softplus(x: torch.Tensor) -> torch.Tensor:
    # log(softplus(x)), clamped for numerical safety
    return F.softplus(x).clamp(min=1e-30).log()


def _log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # log(sigmoid(x)) = -softplus(-x)
    return -F.softplus(-x)


def _log_exp(x: torch.Tensor) -> torch.Tensor:
    # log(exp(x)) = x, clamp to avoid overflow inside pow/softmax later
    return x.clamp(max=20.0)


def _log_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).clamp(min=1e-30).log()


_LOG_SIGMA: Dict[SigmaType, Callable[[torch.Tensor], torch.Tensor]] = {
    "softplus": _log_softplus,
    "sigmoid": _log_sigmoid,
    "exp": _log_exp,
    "relu": _log_relu,
}


def _default_sigma_map() -> Dict[str, SigmaType]:
    return {
        "encoder.self_attn": "softplus",
        "decoder.self_attn": "sigmoid",
        "decoder.encoder_attn": "exp",
    }


def solve_tau_star(
    scores: torch.Tensor,
    sigma: Union[SigmaType, Callable[[torch.Tensor], torch.Tensor]] = "softplus",
    n_iter: int = 30,
    tau_init: float = 2.0,
    tau_min: float = 0.1,
    tau_max: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Newton solver for τ*.

    Args:
        scores: tensor of shape (*, T), last dimension is the attention logits.
        sigma: one of {"softplus", "sigmoid", "exp", "relu"} or a callable
               that returns log(sigma(scores)).
        n_iter: number of Newton steps.
        tau_init: initial τ.
        tau_min / tau_max: clamp bounds.
        eps: damping for zero variance.

    Returns:
        tau: tensor of shape (*)
        kl:  tensor of shape (*)
    """
    if isinstance(sigma, str):
        sigma = _LOG_SIGMA[sigma]

    scores = scores.float()
    shape = scores.shape[:-1]

    # Degenerate case: only one key -> τ is undefined, return default.
    if scores.shape[-1] <= 1:
        tau = torch.full(shape, tau_init, dtype=torch.float32, device=scores.device)
        kl = torch.zeros(shape, dtype=torch.float32, device=scores.device)
        return tau, kl

    log_sigma = sigma(scores)

    # reference distribution = softmax(scores)
    ref = F.softmax(scores, dim=-1)
    e_ref = (ref * log_sigma).sum(dim=-1)

    tau = torch.full(shape, tau_init, dtype=torch.float32, device=scores.device)

    for _ in range(n_iter):
        # p_tau = softmax(tau * log_sigma)
        logits = tau.unsqueeze(-1) * log_sigma
        log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
        p = torch.exp(logits - log_norm)

        e_p = (p * log_sigma).sum(dim=-1)
        var_p = (p * (log_sigma - e_p.unsqueeze(-1)) ** 2).sum(dim=-1)

        grad = e_p - e_ref
        delta = grad / (var_p + eps)

        tau = (tau - delta).clamp(tau_min, tau_max)

    # final KL(ref || p_tau)
    logits = tau.unsqueeze(-1) * log_sigma
    log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
    p = torch.exp(logits - log_norm)
    kl = (ref * (ref / p).clamp(min=1e-30).log()).sum(dim=-1)

    # sanitize degenerate rows (e.g. T=1 after padding)
    tau = torch.where(torch.isfinite(tau), tau, torch.full_like(tau, tau_init))
    kl = torch.where(torch.isfinite(kl), kl, torch.full_like(kl, 0.0))
    return tau, kl


def solve_tau_star_analytic(
    scores: torch.Tensor,
    sigma: Union[SigmaType, Callable[[torch.Tensor], torch.Tensor]] = "softplus",
) -> torch.Tensor:
    """
    One-shot analytic approximation (faster, less accurate):

        tau ≈ Cov_{ref}(scores, log_sigma) / Var_{ref}(log_sigma)

    Useful as a warm-start for the Newton solver.
    """
    if isinstance(sigma, str):
        sigma = _LOG_SIGMA[sigma]
    scores = scores.float()
    log_sigma = sigma(scores)
    ref = F.softmax(scores, dim=-1)

    mu_s = (ref * scores).sum(dim=-1)
    mu_l = (ref * log_sigma).sum(dim=-1)
    cov = (ref * (scores - mu_s.unsqueeze(-1)) * (log_sigma - mu_l.unsqueeze(-1))).sum(dim=-1)
    var = (ref * (log_sigma - mu_l.unsqueeze(-1)) ** 2).sum(dim=-1)
    return (cov / (var + 1e-6)).clamp(0.1, 10.0)


@dataclass
class HeadTau:
    module_name: str
    head_index: int
    tau: float
    kl: float


@dataclass
class ModuleResult:
    module_name: str
    num_heads: int
    tau_per_head: List[float] = field(default_factory=list)
    kl_per_head: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "num_heads": self.num_heads,
            "tau_per_head": self.tau_per_head,
            "kl_per_head": self.kl_per_head,
            "tau_mean": float(sum(self.tau_per_head) / len(self.tau_per_head)) if self.tau_per_head else 0.0,
            "tau_std": _std(self.tau_per_head),
        }


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n))


class TauStarEstimator:
    """
    Collect attention score tensors and estimate τ* per head.

    Example:
        estimator = TauStarEstimator(n_iter=30)
        for scores in collector:               # scores: (B, H, nq, nk)
            estimator.collect("model.encoder.layers.0.self_attn", scores)
        results = estimator.aggregate()
    """

    def __init__(
        self,
        n_iter: int = 30,
        tau_init: float = 2.0,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        sigma_map: Dict[str, SigmaType] | None = None,
    ):
        self.n_iter = n_iter
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.sigma_map = sigma_map or _default_sigma_map()
        self.buffers: Dict[str, List[torch.Tensor]] = {}

    def _resolve_sigma(self, module_name: str) -> SigmaType:
        for key, val in self.sigma_map.items():
            if key in module_name:
                return val
        return "softplus"

    def collect(self, module_name: str, scores: torch.Tensor):
        """
        scores: (B, H, nq, nk) or (B, H, T) or any shape whose last dim is key length
                  and second-to-last dim is number of heads.
        """
        if scores.numel() == 0:
            return
        # keep on CPU to avoid GPU memory bloat; move to GPU only at solve time
        self.buffers.setdefault(module_name, []).append(scores.detach().cpu())

    def aggregate(self, device: str = "cuda") -> List[ModuleResult]:
        results = []
        for module_name, tensors in self.buffers.items():
            # flatten to (N, H, T) and pad variable key lengths
            flat = []
            n_heads = None
            t_max = 0
            for t in tensors:
                # Flatten batch & query dims, keep head dim: (N, H, T)
                t = t.reshape(-1, t.shape[1], t.shape[-1])
                flat.append(t)
                n_heads = t.shape[1]
                t_max = max(t_max, t.shape[2])

            padded = []
            for t in flat:
                if t.shape[2] < t_max:
                    pad = torch.full(
                        (t.shape[0], n_heads, t_max - t.shape[2]),
                        -1e9,
                        dtype=torch.float32,
                    )
                    t = torch.cat([t, pad], dim=2)
                padded.append(t)
            scores = torch.cat(padded, dim=0).float().to(device)
            sigma = self._resolve_sigma(module_name)

            tau, kl = solve_tau_star(
                scores,
                sigma=sigma,
                n_iter=self.n_iter,
                tau_init=self.tau_init,
                tau_min=self.tau_min,
                tau_max=self.tau_max,
            )
            tau = tau.cpu()
            kl = kl.cpu()

            mod_res = ModuleResult(module_name=module_name, num_heads=n_heads)
            for h in range(n_heads):
                mod_res.tau_per_head.append(round(float(tau[h].mean()), 4))
                mod_res.kl_per_head.append(round(float(kl[h].mean()), 6))
            results.append(mod_res)
        return results

    def save(self, path: str, results: List[ModuleResult] | None = None):
        if results is None:
            results = self.aggregate()
        payload = [r.to_dict() for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # quick self-test
    torch.manual_seed(0)
    scores = torch.randn(8, 12, 256) * 2.0
    tau, kl = solve_tau_star(scores, sigma="softplus", n_iter=20)
    print("tau mean:", tau.mean().item())
    print("kl mean:", kl.mean().item())
