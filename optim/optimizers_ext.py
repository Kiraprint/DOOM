"""AdamW and Muon for Sample Factory 2.1.1, plus the patch that installs them.

Why this file exists: Sample Factory builds its optimizer with

    optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)
    optimizer_kwargs = dict(lr=..., betas=...)
    self.optimizer = optimizer_cls(params, **optimizer_kwargs)

so `weight_decay` is never passed: every `--weight_decay` value ever recorded in
this project's config.json was ignored, and `--optimizer adamw` is not even an
accepted choice (`choices=["adam", "lamb"]`), which is why 39 of the 44 HPO
trials that sampled adamw died in 5 s.

`install()` rewrites the one constructor block inside the installed
`sample_factory/algo/learning/learner.py` so the same cfg can select adam /
adamw / lamb / muon, and so `cfg.weight_decay` finally reaches the optimizer.
A monkeypatch cannot work here: Sample Factory runs the Learner in a spawned
worker that re-imports the module and loses in-process patches. `describe()`
reports the file and both digests, so the provenance stays citable.

Muon (Jordan et al., "Muon: An optimizer for hidden layers in neural networks",
2024) applies a Newton-Schulz orthogonalisation to the update of every 2D weight
matrix and falls back to AdamW for embeddings, biases, norms and 1D parameters.
Scale convention used here: the orthogonalised update is multiplied by
sqrt(max(1, rows/cols)) * 0.2, the reference implementation's choice.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch

LOG = logging.getLogger(__name__)

MUON_NAMES = ("adam", "adamw", "lamb", "muon")


# --------------------------------------------------------------------------- Muon


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalise a 2D matrix by quintic Newton-Schulz iteration.

    Runs in float32 regardless of the parameter dtype: the iteration squares the
    matrix, so bf16 loses the small singular values that make the step useful.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden-layer weights, AdamW for everything else.

    Parameters are partitioned by shape, not by name, so the caller does not have
    to know Sample Factory's module layout: a parameter participates in the Muon
    group only when `use_muon(p)` says it is a genuine matrix (2D and both
    dimensions above `min_dim`). Anything else goes to the AdamW group verbatim.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        fallback_lr: Optional[float] = None,
        min_dim: int = 1,
    ):
        matrix: List[torch.nn.Parameter] = []
        other: List[torch.nn.Parameter] = []
        for p in params:
            if not p.requires_grad:
                continue
            (matrix if _is_muonable(p, min_dim) else other).append(p)

        # The two groups must not share a learning rate: Muon's orthogonalised
        # update has a different scale from an Adam step, so running the AdamW
        # fallback at the Muon lr drives embeddings/biases/norms at that lr.
        # The first Muon arm did exactly that (lr 0.02 = 50x this campaign's
        # tuned Adam lr of 4.05e-4) and never learned; `fallback_lr` separates
        # the two knobs.
        other_lr = lr if fallback_lr is None else float(fallback_lr)
        groups = []
        if matrix:
            groups.append(dict(params=matrix, use_muon=True, lr=lr))
        if other:
            groups.append(dict(params=other, use_muon=False, lr=other_lr))
        if not groups:
            raise ValueError("Muon got no trainable parameters")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=adamw_betas,
            eps=adamw_eps,
            use_muon=False,
        )
        super().__init__(groups, defaults)
        LOG.info(
            "Muon: %d matrix tensors (%d params) orthogonalised at lr %.3g, "
            "%d other tensors (%d params) via AdamW at lr %.3g, wd %.3g",
            len(matrix),
            sum(p.numel() for p in matrix),
            lr,
            len(other),
            sum(p.numel() for p in other),
            other_lr,
            weight_decay,
        )

    def zero_grad(self, set_to_none: bool = True):
        """Honour `set_to_none`; Sample Factory calls this with no arguments."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: Dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            # Decoupled decay on the matrix group too. Without this the arm
            # labelled "+ wd" applied decay only to the AdamW fallback group,
            # i.e. only to biases/embeddings, which is not the arm advertised.
            if wd:
                p.mul_(1 - lr * wd)
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

            if update.ndim != 2:
                raise RuntimeError(f"Muon update must be 2D, got shape {tuple(update.shape)}")
            orth = _zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
            scale = max(1.0, orth.size(0) / orth.size(1)) ** 0.5
            p.add_(orth.to(p.dtype), alpha=-lr * scale * 0.2)

    def _step_adamw(self, group: Dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon's AdamW branch does not support sparse gradients")
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            # Decoupled decay: shrink the weights, not the update.
            if wd:
                p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bc1)


def _is_muonable(p: torch.nn.Parameter, min_dim: int) -> bool:
    return p.ndim == 2 and min(p.shape) > min_dim


# ------------------------------------------------------------------- installation


def build_optimizer(cfg, params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """Construct the optimizer named by cfg.optimizer with cfg.weight_decay applied.

    Mirrors Sample Factory's own construction so the only difference is the one
    under study: same lr, same betas, same eps. This is the single place where
    `cfg.weight_decay` is read, which is what makes the knob real.
    """
    name = str(cfg.optimizer).lower()
    lr = cfg.learning_rate
    betas = (cfg.adam_beta1, cfg.adam_beta2)
    eps = cfg.adam_eps
    wd = float(getattr(cfg, "weight_decay", 0.0) or 0.0)

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    if name == "lamb":
        from sample_factory.algo.utils.optimizers import Lamb

        return Lamb(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    if name == "muon":
        # cfg.muon_lr is the orthogonalised-update lr; cfg.learning_rate stays the
        # AdamW fallback lr so the arm differs from the Adam arms in exactly one
        # respect. When --muon_lr is absent both use cfg.learning_rate (the
        # behaviour of the first, broken Muon arm).
        muon_lr = float(getattr(cfg, "muon_lr", 0.0) or lr)
        return Muon(
            params,
            lr=muon_lr,
            weight_decay=wd,
            adamw_betas=betas,
            adamw_eps=eps,
            fallback_lr=lr,
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected one of {MUON_NAMES}")


ANCHOR_START = "        optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)\n"
ANCHOR_END = "        self.optimizer = optimizer_cls(params, **optimizer_kwargs)\n"
MARKER = "# --- optimizer patch (optimizers_ext) ---"

REPLACEMENT = """        # --- optimizer patch (optimizers_ext) ---
        # Stock Sample Factory accepts only adam/lamb here and never passes
        # weight_decay to either, so cfg.weight_decay had no effect on any run
        # recorded before this patch. Both are fixed by delegating to the
        # optimizer table in optimizers_ext, which is imported by path because
        # this file lives in site-packages.
        from optimizers_ext import build_optimizer

        self.optimizer = build_optimizer(self.cfg, params)
"""


def install(learner_path: Optional[str] = None) -> Dict[str, object]:
    """Make Sample Factory build its optimizer through `build_optimizer`.

    A monkeypatch is not enough: Sample Factory runs the Learner in a spawned
    worker process, which re-imports `learner.py` from scratch and loses any
    in-process patch (observed as `RuntimeError: Unknown optimizer adamw` inside
    the worker while the parent had reported a successful patch). This rewrites
    the single constructor block in the installed file instead, which holds for
    fork and spawn alike.

    Idempotent, reversible (a `.bak_optimizers_ext` copy is left beside it), and
    it refuses to touch anything unless the anchor text matches exactly once.
    """
    import importlib.util
    from pathlib import Path

    if learner_path is None:
        spec = importlib.util.find_spec("sample_factory.algo.learning.learner")
        learner_path = spec.origin if spec and spec.origin else None
    if not learner_path:
        raise RuntimeError("cannot locate sample_factory/algo/learning/learner.py")

    path = Path(learner_path)
    original = path.read_text()
    before = hashlib.sha256(original.encode()).hexdigest()[:16]

    record: Dict[str, object] = {"path": str(path), "sha256_before": before}
    if MARKER in original:
        record.update(patched=False, reason="already patched", sha256_after=before)
        LOG.info("optimizer patch already present in %s", path)
        return record

    if original.count(ANCHOR_START) != 1 or original.count(ANCHOR_END) != 1:
        raise RuntimeError(
            f"anchor not found exactly once in {path}: "
            f"{original.count(ANCHOR_START)} start / {original.count(ANCHOR_END)} end"
        )

    start = original.index(ANCHOR_START)
    end = original.index(ANCHOR_END) + len(ANCHOR_END)
    patched = original[:start] + REPLACEMENT + original[end:]

    backup = path.with_suffix(path.suffix + ".bak_optimizers_ext")
    if not backup.exists():
        backup.write_text(original)
    path.write_text(patched)

    after = hashlib.sha256(patched.encode()).hexdigest()[:16]
    record.update(
        patched=True,
        sha256_after=after,
        backup=str(backup),
        removed_lines=original[start:end].count("\n"),
        added_lines=REPLACEMENT.count("\n"),
    )
    LOG.warning("optimizer patch written to %s (sha256 %s -> %s)", path, before, after)
    return record


def describe(record: Optional[Dict[str, object]] = None) -> str:
    r = record or {}
    # `patched=False` means install() found the marker already in place, i.e. the
    # file IS patched and the digests are equal. Printing the raw flag read as
    # "not patched" in every config.json of the grid, which would misrepresent the
    # runs, so the state is spelled out.
    state = "applied-now" if r.get("patched") else ("already-applied" if r.get("reason") == "already patched" else "not-applied")
    if "sha256_before" in r:
        return "optimizer patch: {state} over {path} sha256 {b} -> {a}".format(
            state=state, path=r.get("path"),
            b=r.get("sha256_before"), a=r.get("sha256_after", "?"),
        )
    return "optimizer patch: {state} over {path}".format(state=state, path=r.get("path", "?"))
