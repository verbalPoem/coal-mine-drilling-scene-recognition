from contextlib import nullcontext
import threading
import torch

_active = threading.local()


class NonFiniteComputation(RuntimeError):
    pass


def floating_tensors(value, prefix="value"):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            yield prefix, value
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from floating_tensors(item, f"{prefix}[{i}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from floating_tensors(item, f"{prefix}[{key!r}]")


def map_floating(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(map_floating(item, dtype) for item in value)
    if isinstance(value, list):
        return [map_floating(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: map_floating(item, dtype) for key, item in value.items()}
    return value


def assert_finite(value, where):
    for path, tensor in floating_tensors(value):
        if not bool(torch.isfinite(tensor).all().item()):
            raise NonFiniteComputation(
                f"{where}: nonfinite tensor {path}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
            )


def _make_fp32_forward(cls, original):
    def forward(self, *args, **kwargs):
        active = getattr(_active, "modules", set())
        if id(self) in active:
            return original(self, *args, **kwargs)
        floating = [tensor for _, tensor in floating_tensors((args, kwargs))]
        if not floating:
            return original(self, *args, **kwargs)
        assert_finite((args, kwargs), cls.__name__ + " input")
        device = floating[0].device.type
        return_dtype = floating[0].dtype
        if any(t.device.type != device for t in floating):
            raise RuntimeError("Mixed-device protected forward is unsupported")
        if any(t.dtype != return_dtype for t in floating):
            raise RuntimeError(
                "Mixed floating input dtypes require an explicit output policy"
            )
        context = (
            torch.autocast(device_type=device, enabled=False)
            if device in {"cuda", "cpu"}
            else nullcontext()
        )
        args32, kwargs32 = (
            map_floating(args, torch.float32),
            map_floating(kwargs, torch.float32),
        )
        with context:
            state = dict(self.named_parameters())
            state.update(dict(self.named_buffers()))
            needs_temporary_state = any(
                t.is_floating_point() and t.dtype != torch.float32
                for t in state.values()
            )
            if needs_temporary_state:
                # Trainer validation casts the EMA model to half. Temporary
                # functional state handles that without mutating its storage.
                state32 = {
                    key: tensor.float() if tensor.is_floating_point() else tensor
                    for key, tensor in state.items()
                }
                _active.modules = active | {id(self)}
                try:
                    output = torch.func.functional_call(
                        self, state32, args32, kwargs32, strict=True
                    )
                finally:
                    _active.modules = active
            else:
                output = original(self, *args32, **kwargs32)
        assert_finite(output, cls.__name__ + " FP32 output")
        if return_dtype in {torch.float16, torch.bfloat16}:
            maximum = torch.finfo(return_dtype).max
            for path, tensor in floating_tensors(output):
                if (
                    tensor.numel()
                    and float(tensor.detach().abs().max().item()) > maximum
                ):
                    raise NonFiniteComputation(
                        f"{cls.__name__}: finite FP32 output cannot be represented as {return_dtype} at {path}; no clipping performed"
                    )
        converted = map_floating(output, return_dtype)
        assert_finite(converted, cls.__name__ + " returned output")
        return converted

    forward.__name__ = "forward"
    forward.__qualname__ = cls.__qualname__ + ".forward"
    return forward
