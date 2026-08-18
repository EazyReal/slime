"""Scale policy logits to match rollout-time temperature."""

import torch


def scale_logits_by_rollout_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide policy logits by rollout temperature when it is a positive non-1 scale.

    Temperature 0 is greedy decoding and must not inf-out the logits. Value heads
    (last dim 1) are left unchanged.
    """
    if logits.size(-1) > 1 and temperature > 0 and temperature != 1.0:
        return logits / temperature
    return logits
