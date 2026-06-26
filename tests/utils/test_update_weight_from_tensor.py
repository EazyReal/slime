import importlib
import sys
import types

import pytest
import torch


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return "ref"


class _Engine:
    def __init__(self):
        self.update_weights_from_tensor = _RemoteMethod()


class _Dist:
    def __init__(self, gathered=None):
        self.gathered = gathered

    def get_rank(self):
        return 0

    def get_world_size(self, _group):
        return len(self.gathered) if self.gathered is not None else 1

    def gather_object(self, obj, object_gather_list, dst, group):
        object_gather_list[:] = self.gathered if self.gathered is not None else [obj]


def _load_update_weight_from_tensor(monkeypatch):
    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_tensor"
    distributed_module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    sys.modules.pop(module_name, None)
    sys.modules.pop(distributed_module_name, None)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    actor_mod = types.ModuleType("ray.actor")
    actor_mod.ActorHandle = object

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    mpu_mod = types.ModuleType("megatron.core.mpu")
    core_mod.mpu = mpu_mod
    megatron_mod.core = core_mod

    sglang_mod = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_mod.FlattenedTensorBucket = object
    sglang_mod.MultiprocessingSerializer = object
    sglang_mod.DeltaSpec = object

    distributed_mod = types.ModuleType(distributed_module_name)
    distributed_mod.connect_rollout_engines_from_distributed = object
    distributed_mod.disconnect_rollout_engines_from_distributed = object
    distributed_mod.post_process_weights = object
    distributed_mod.update_weights_from_distributed = object

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, distributed_module_name, distributed_mod)
    return importlib.import_module(module_name)


def _install_bucket_fakes(monkeypatch, module):
    bucket_calls = []

    class Bucket:
        supports_multi_dtypes = True

        def __init__(self, named_tensors):
            if not named_tensors:
                raise ValueError("empty")
            bucket_calls.append(named_tensors)
            self.named_tensors = named_tensors

        def get_metadata(self):
            return {"names": [name for name, _ in self.named_tensors]}

        def get_flattened_tensor(self):
            return "flattened"

    def serialize(value, output_str):
        assert output_str is True
        return ",".join(value["metadata"]["names"])

    monkeypatch.setattr(module, "FlattenedTensorBucket", Bucket)
    monkeypatch.setattr(module, "MultiprocessingSerializer", types.SimpleNamespace(serialize=serialize))
    return bucket_calls


def test_send_to_colocated_engine_skips_empty_bucket(monkeypatch):
    module = _load_update_weight_from_tensor(monkeypatch)
    bucket_calls = _install_bucket_fakes(monkeypatch, module)
    monkeypatch.setattr(module, "dist", _Dist())
    engine = _Engine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=1,
    )

    assert refs == []
    assert long_lived_tensors == []
    assert bucket_calls == []
    assert engine.update_weights_from_tensor.calls == []


def test_send_to_colocated_engine_sends_non_empty_bucket(monkeypatch):
    module = _load_update_weight_from_tensor(monkeypatch)
    bucket_calls = _install_bucket_fakes(monkeypatch, module)
    monkeypatch.setattr(module, "dist", _Dist())
    engine = _Engine()
    tensor = torch.ones(2, dtype=torch.float32)

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [("layer.weight", tensor)],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == ["ref"]
    assert len(long_lived_tensors) == 1
    assert bucket_calls == [[("layer.weight", tensor)]]
    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": ["layer.weight"],
            "load_format": "flattened_bucket",
            "weight_version": "7",
        }
    ]


def test_send_to_colocated_engine_rejects_mismatched_bucket_counts(monkeypatch):
    module = _load_update_weight_from_tensor(monkeypatch)
    _install_bucket_fakes(monkeypatch, module)
    monkeypatch.setattr(module, "dist", _Dist(gathered=[[], ["layer.weight"]]))

    with pytest.raises(RuntimeError, match="expected equal bucket counts"):
        module._send_to_colocated_engine(
            [],
            ipc_engine=_Engine(),
            ipc_gather_src=0,
            ipc_gather_group=object(),
            weight_version=1,
        )
