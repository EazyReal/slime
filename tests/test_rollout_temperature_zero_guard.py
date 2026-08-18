import torch

from slime.utils.logit_scale import scale_logits_by_rollout_temperature

NUM_GPUS = 0


def test_zero_temperature_leaves_policy_logits_unchanged():
    logits = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    out = scale_logits_by_rollout_temperature(logits, 0.0)
    torch.testing.assert_close(out, logits)


def test_nonzero_temperature_divides_policy_logits():
    logits = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    out = scale_logits_by_rollout_temperature(logits, 0.5)
    torch.testing.assert_close(out, logits / 0.5)


def test_unit_temperature_is_a_no_op():
    logits = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    out = scale_logits_by_rollout_temperature(logits, 1.0)
    torch.testing.assert_close(out, logits)


def test_value_head_is_never_scaled():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    out = scale_logits_by_rollout_temperature(values, 0.5)
    torch.testing.assert_close(out, values)
