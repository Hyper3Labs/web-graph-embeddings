"""Edge scoring + binary cross-entropy losses.

``_inbatch_loss`` is the published objective: dot scores against ``--train-neg`` uniformly
corrupted destinations, BCE on positives and negatives independently. Both losses take the edge
``decoder`` from :mod:`cupyg.geometry` (default: the published dot product), so ``--decoder
hyperbolic`` reuses the identical objective with ``(margin - d) / temperature`` logits.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from cupyg.geometry import DOT, Decoder


def _inbatch_loss(
    x: torch.Tensor, batch: object, train_neg: int, decoder: Decoder = DOT
) -> tuple[torch.Tensor, int]:
    """Binary-NLL loss with ``train_neg`` manual negatives per positive (in-batch corruption).

    The cugraph sampler crashes with ``neg_sampling`` amount>1, so to score more than one negative
    per positive we ignore the loader's built-in negative and corrupt each positive's destination
    ourselves: sample ``train_neg`` random *local* nodes from the already-encoded sampled subgraph
    (``x``, the node reps from ``model.forward``). This works for both encoders (the nodes are
    in ``x``, so SAGE's message-passed representations are reused) and biases negatives toward
    structurally-nearby nodes (harder, and the leakage risk of hitting a true neighbour is tiny).
    """
    eli = batch.edge_label_index
    pos_mask = batch.edge_label.to(x.device) == 1
    src, dst = eli[0][pos_mask], eli[1][pos_mask]
    hs = x[src]
    pos_score = decoder(hs, x[dst])
    neg_idx = torch.randint(0, x.shape[0], (src.shape[0], train_neg), device=x.device)
    neg_score = decoder(hs.unsqueeze(1), x[neg_idx])
    pos_loss = functional.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = functional.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss, pos_score.numel() + neg_score.numel()



def _binary_loss(
    x: torch.Tensor, batch: object, decoder: Decoder = DOT
) -> tuple[torch.Tensor, int]:
    """Binary-NLL over the loader's single on-GPU negative per positive (``edge_label`` targets)."""
    eli = batch.edge_label_index
    out = decoder(x[eli[0]], x[eli[1]])
    loss = functional.binary_cross_entropy_with_logits(out, batch.edge_label.to(x.device))
    return loss, out.numel()
