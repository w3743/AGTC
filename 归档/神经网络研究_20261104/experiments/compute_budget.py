"""Auditable, execution-based scalar-operation proxy for PyTorch models.

Usage::

    from compute_budget import ArithmeticCounter
    counter = ArithmeticCounter()
    with counter:
        prediction = model(x)
        loss = loss_fn(prediction, target)
        loss.backward()                 # include training backward in same trace
    print(counter.stats_json())

This is a software-operation proxy, not hardware FLOPs or latency. Matrix
products count 2 scalar operations per multiply-accumulate. Elementwise
arithmetic counts its constituent scalar operations per output element. Simple
transcendental functions use one proxy operation per element. Softmax and layer
normalization use the explicit formulas documented by ``_special_cost``;
unknown aten operators are reported in
``unsupported`` and never silently treated as zero. Operators that only
describe/manipulate tensor metadata, copy data, or select/index values are in
the explicit ``zero_cost_allowlist``; that choice means their arithmetic
work is outside this arithmetic-only proxy.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _base_name(func: Any) -> str:
    schema = getattr(func, "_schema", None)
    return schema.name if schema is not None else str(func).split(".")[0]


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _first_tensor(value: Any):
    return next(_tensors(value), None)


def _numel(t: torch.Tensor) -> int:
    return int(t.numel())


# Names are aten schema names (without overload suffix). Keep this list
# intentionally visible so exclusions can be audited and extended explicitly.
ZERO_COST_ALLOWLIST = frozenset({
    # Shape, stride, view, dtype/device and other metadata operations.
    "aten::alias", "aten::as_strided", "aten::detach", "aten::expand",
    "aten::flatten", "aten::movedim", "aten::permute", "aten::reshape",
    "aten::resolve_conj", "aten::resolve_neg", "aten::select",
    "aten::slice", "aten::squeeze", "aten::t", "aten::transpose", "aten::unsqueeze",
    "aten::view", "aten::view_as", "aten::_unsafe_view", "aten::contiguous",
    "aten::is_same_size", "aten::sym_size", "aten::sym_stride",
    "aten::size", "aten::stride", "aten::numel", "aten::dim",
    "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::zeros",
    "aten::ones", "aten::new_empty", "aten::new_zeros", "aten::new_empty_strided",
    # Explicit data movement / indexing / selection, not arithmetic in this proxy.
    "aten::copy_", "aten::to", "aten::_to_copy", "aten::lift_fresh",
    "aten::lift_fresh_copy", "aten::index", "aten::index_select",
    "aten::gather", "aten::scatter", "aten::scatter_", "aten::where",
    "aten::masked_select", "aten::masked_fill", "aten::masked_fill_",
    "aten::fill_", "aten::select_scatter", "aten::slice_scatter",
    "aten::unbind", "aten::split", "aten::split_with_sizes", "aten::chunk",
    "aten::cat", "aten::stack", "aten::narrow", "aten::unbind_copy",
    "aten::ones_like", "aten::zeros_like", "aten::empty_permuted",
    "aten::clone", "aten::constant_pad_nd", "aten::unfold", "aten::triu",
    "aten::slice_backward", "profiler::_record_function_enter_new",
    "profiler::_record_function_exit",
    # Bookkeeping / autograd wrappers with no numerical transform.
    "aten::detach_", "aten::set_requires_grad", "aten::alias_copy",
    "aten::_local_scalar_dense", "custom::_local_scalar_dense", "custom::_local_scalar_dense64",
})


_ELEMENTWISE = frozenset({
    "aten::add", "aten::add_", "aten::sub", "aten::sub_", "aten::mul", "aten::mul_", "aten::div", "aten::div_", "aten::true_divide",
    "aten::floor_divide", "aten::pow", "aten::neg", "aten::abs", "aten::rsub", "aten::rsub_", "custom::rsub",
    "aten::reciprocal", "aten::square",
    "aten::maximum", "aten::minimum", "aten::clamp", "aten::clamp_", "aten::clamp_min",
    "aten::clamp_min_", "aten::clamp_max", "aten::clamp_max_", "aten::lerp", "aten::lerp_", "aten::addcmul", "aten::addcmul_", "aten::addcdiv", "aten::addcdiv_",
    "aten::fma", "aten::remainder", "aten::fmod", "aten::sign",
    "aten::threshold", "aten::threshold_backward", "aten::relu",
    "aten::relu_", "aten::relu_backward", "aten::gelu_backward", "aten::silu_backward",
    "aten::elu", "aten::elu_backward", "aten::leaky_relu",
    "aten::leaky_relu_backward", "aten::hardtanh", "aten::hardtanh_backward",
    "aten::prelu", "aten::prelu_backward", "aten::eq", "aten::ne",
    "aten::lt", "aten::le", "aten::gt", "aten::ge", "aten::logical_and",
    "aten::logical_or", "aten::logical_not", "aten::bitwise_and",
    "aten::bitwise_or", "aten::bitwise_xor", "aten::bitwise_not",
    "aten::isnan", "aten::isinf", "aten::isfinite", "aten::isclose",
    "aten::exp", "aten::expm1", "aten::log", "aten::log1p", "aten::log2",
    "aten::log10", "aten::sin", "aten::cos", "aten::tan", "aten::tanh",
    "aten::sigmoid", "aten::silu", "aten::gelu", "aten::erf", "aten::floor",
    "aten::ceil", "aten::round", "aten::trunc", "aten::frac", "aten::nan_to_num",
    "aten::dropout", "aten::native_dropout", "aten::bernoulli",
    "aten::bernoulli_", "aten::uniform_", "aten::normal_",
})

_SIMPLE_NONLINEAR = frozenset({
    "aten::tanh", "aten::sigmoid", "aten::gelu", "aten::silu",
})
_SPECIAL = frozenset({
    "aten::sqrt", "aten::rsqrt", "aten::softmax", "aten::_softmax",
    "aten::log_softmax", "aten::_log_softmax", "aten::tanh_backward",
    "aten::sigmoid_backward", "aten::relu_backward", "aten::threshold_backward",
    "aten::softplus", "aten::softplus_backward", "custom::softplus",
    "aten::_softmax_backward_data", "aten::_log_softmax_backward_data",
    "aten::native_layer_norm", "aten::native_layer_norm_backward", "aten::layer_norm",
    "aten::linalg_vector_norm", "aten::nll_loss_forward", "aten::nll_loss_backward",
    "aten::nll_loss2d_forward", "aten::nll_loss2d_backward", "aten::_foreach_norm",
})

_REDUCTIONS = frozenset({
    "aten::sum", "aten::mean", "aten::sum_dim_IntList", "aten::mean_dim",
    "aten::sum_IntList", "aten::mean.names_dim", "aten::prod",
    "aten::amax", "aten::amin", "aten::max", "aten::min", "aten::argmax",
    "aten::argmin", "aten::all", "aten::any", "aten::var", "aten::std",
    "aten::var_mean", "aten::std_mean", "aten::logsumexp",
    "aten::aminmax", "custom::aminmax",
})

_MATMUL = frozenset({"aten::mm", "aten::bmm", "aten::matmul", "aten::mv", "aten::dot", "aten::addmm", "aten::addmv"})


class ArithmeticCounter(TorchDispatchMode):
    """Count observed aten operations in an inference/training execution scope."""

    def __init__(self):
        super().__init__()
        self.arithmetic = 0
        self.nonlinear = 0
        self.special = 0
        self.per_op = Counter()
        self.unsupported = Counter()
        self.zero_cost = Counter()
        self.examples = defaultdict(list)

    def _record(self, bucket: Counter, name: str, amount: int, tensors=()):
        bucket[name] += int(amount)
        key = name
        if len(self.examples[key]) < 4:
            shapes = [list(t.shape) for t in tensors if isinstance(t, torch.Tensor)]
            sample = {"shapes": shapes, "count": int(amount)}
            if sample not in self.examples[key]:
                self.examples[key].append(sample)

    @staticmethod
    def _matmul_cost(name: str, args, out) -> int | None:
        # addmm/addmv carry a destination/bias as argument zero; the actual
        # product operands are arguments one and two.
        offset = 1 if name in ("aten::addmm", "aten::addmv") else 0
        a, b = _first_tensor(args[offset]), _first_tensor(args[offset + 1])
        if a is None or b is None or a.dim() < 1 or b.dim() < 1:
            return None
        # Matrix multiply output has each value formed by K multiply-add pairs.
        if a.dim() == 1 and b.dim() == 1:
            k, outputs = int(a.shape[0]), 1
        elif a.dim() == 1:
            k, outputs = int(a.shape[-1]), _numel(_first_tensor(out))
        elif b.dim() == 1:
            k, outputs = int(b.shape[0]), _numel(_first_tensor(out))
        else:
            k = int(a.shape[-1])
            outputs = _numel(_first_tensor(out))
        return 2 * outputs * k

    @staticmethod
    def _special_cost(name: str, args, out) -> int | None:
        """Return (scalar arithmetic proxy) for explicitly modeled operators.

        Softmax over groups of width K uses exp K + max-subtract K + reduction
        adds K-1 + divide K = 4K-1 per group. Its backward uses the standard
        y*(g-sum(g*y)) form: 4K-1. Log-softmax adds a log and a final subtract,
        yielding 4K per group forward; backward uses exp+reduce+sub+mul = 4K-1.

        Layer norm forward counts mean/reduction, centering, squared deviation,
        variance reduction, inverse standard deviation, normalization and
        optional affine scale/shift: 5K+1 plus K for each present affine tensor
        per normalized group. Backward counts the two normalized-group
        reductions, input-gradient arithmetic, and optional scale/shift
        gradients using a transparent per-input proxy. Softplus uses
        log1p(exp(beta*x)) and beta scaling/division where beta differs from 1.
        These estimates are not kernel FLOPs.
        """
        x = _first_tensor(args[0]) if args else None
        y = _first_tensor(out)
        n = _numel(x) if x is not None else 0
        nout = _numel(y) if y is not None else 0
        if name == "aten::_foreach_norm":
            # L2 norm per tensor: N squares + (N-1) adds + one sqrt = 2N.
            return 2 * sum(_numel(t) for t in _tensors(args[0]))
        if name in ("aten::softmax", "aten::_softmax", "aten::log_softmax", "aten::_log_softmax",
                    "aten::_softmax_backward_data", "aten::_log_softmax_backward_data"):
            # Forward input is arg0; backward input is grad_output (arg0), and
            # the normalized tensor is arg1. Shapes are equal for both.
            source = _first_tensor(args[1]) if "backward" in name and len(args) > 1 else x
            if source is None:
                return None
            axis = int(args[1] if name in ("aten::softmax", "aten::_softmax", "aten::log_softmax", "aten::_log_softmax") else args[2])
            axis %= source.dim()
            width = int(source.shape[axis])
            groups = _numel(source) // max(1, width)
            if "backward" in name:
                return groups * (4 * width - 1)
            if "log_softmax" in name:
                return groups * (4 * width)
            return groups * (4 * width - 1)
        if name in ("aten::tanh_backward", "aten::sigmoid_backward"):
            return 3 * nout  # (1-y*y)*grad or y*(1-y)*grad
        if name in ("aten::relu_backward", "aten::threshold_backward"):
            return nout  # one mask/select multiply proxy
        if name == "aten::native_layer_norm":
            if x is None or len(args) < 2:
                return None
            norm_shape = args[1]
            width = 1
            for d in norm_shape:
                width *= int(d)
            groups = n // max(1, width)
            has_weight = len(args) > 2 and args[2] is not None
            has_bias = len(args) > 3 and args[3] is not None
            return groups * (5 * width + 1 + width * (int(has_weight) + int(has_bias)))
        if name == "aten::layer_norm":
            # Functional wrapper may dispatch as a single op on some backends.
            if x is None or len(args) < 2:
                return None
            width = 1
            for d in args[1]:
                width *= int(d)
            groups = n // max(1, width)
            has_weight = len(args) > 2 and args[2] is not None
            has_bias = len(args) > 3 and args[3] is not None
            return groups * (5 * width + 1 + width * (int(has_weight) + int(has_bias)))
        if name == "aten::native_layer_norm_backward":
            # args: grad_out, input, normalized_shape, mean, rstd, weight, ...
            if len(args) < 3:
                return None
            inp = _first_tensor(args[1])
            if inp is None:
                return None
            width = 1
            for d in args[2]:
                width *= int(d)
            total = _numel(inp)
            groups = total // max(1, width)
            has_weight = len(args) > 5 and args[5] is not None
            has_bias = len(args) > 6 and args[6] is not None
            # Per input: dyg, dyg*xhat, and dx's 2 subtracts + scale;
            # optional dw multiply; include reductions for affine parameter
            # gradients across the outer groups when weight is present.
            base = total * (6 if has_weight else 4)
            within_group = groups * (2 * max(0, width - 1) + 2)
            affine_reductions = width * max(0, groups - 1) * (int(has_weight) + int(has_bias))
            return base + within_group + affine_reductions
        if name == "aten::linalg_vector_norm":
            if x is None:
                return None
            order = args[1] if len(args) > 1 else 2
            if order != 2:
                return None
            return n + max(0, n - nout)  # squares and reduction additions
        if name in ("aten::nll_loss_forward", "aten::nll_loss2d_forward"):
            target = _first_tensor(args[1]) if len(args) > 1 else None
            if target is None:
                return None
            count = _numel(target)  # gather is excluded; negate is one per target
            reduction = int(args[3]) if len(args) > 3 and args[3] is not None else 1
            if reduction in (1, 2):
                count += max(0, count - 1)
            if reduction == 1 and count:
                count += 1  # mean normalization
            return count
        if name in ("aten::nll_loss_backward", "aten::nll_loss2d_backward"):
            target = _first_tensor(args[2]) if len(args) > 2 else None
            if target is None:
                return None
            count = _numel(target)  # grad_output scale per selected target
            reduction = int(args[4]) if len(args) > 4 and args[4] is not None else 1
            if reduction == 1:
                count *= 2  # normalize by total weight, then scale by grad_output
            return count
        if name in ("aten::sqrt", "aten::rsqrt"):
            return nout
        if name in ("aten::softplus", "custom::softplus"):
            beta = args[1] if len(args) > 1 else 1
            # log1p(exp(beta*x)); include beta scaling when non-unit.
            return nout * (2 if beta == 1 else 4)  # beta multiply + final beta divide
        if name == "aten::softplus_backward":
            beta = args[2] if len(args) > 2 else 1
            # sigmoid(beta*x)*grad: negate, exp, add-one, divide, multiply.
            return nout * (5 if beta == 1 else 6)
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = _base_name(func)
        out_t = _first_tensor(out)
        nout = _numel(out_t) if out_t is not None else 0
        if name.startswith("aten::_foreach_"):
            nout = sum(_numel(t) for t in _tensors(out))
        ins = list(_tensors(args))

        amount = None
        category = "arithmetic"
        if name in _MATMUL:
            amount = self._matmul_cost(name, args, out)
            if amount is not None and name in ("aten::addmm", "aten::addmv"):
                beta = kwargs.get("beta", 1)
                alpha = kwargs.get("alpha", 1)
                if beta != 0:
                    amount += nout  # add scaled bias/residual
                    if beta != 1:
                        amount += nout
                if alpha != 1:
                    amount += nout
        elif name in ("aten::convolution", "aten::_convolution"):
            x = _first_tensor(args[0])
            weight = _first_tensor(args[1])
            if x is not None and weight is not None and out_t is not None:
                groups = int(args[8]) if len(args) > 8 and isinstance(args[8], int) else 1
                kernel = 1
                for d in weight.shape[2:]:
                    kernel *= int(d)
                amount = 2 * nout * int(weight.shape[1]) * kernel
                if len(args) > 2 and args[2] is not None:
                    amount += nout
        elif name in _SPECIAL:
            amount = self._special_cost(name, args, out)
            category = "special"
        elif name in _SIMPLE_NONLINEAR:
            amount = nout
            category = "nonlinear"
        elif name in _ELEMENTWISE:
            if name in ("aten::lerp", "aten::lerp_"):
                amount = 3 * nout
            elif name in ("aten::addcmul", "aten::addcmul_", "aten::addcdiv", "aten::addcdiv_"):
                amount = 3 * nout
            elif name in ("aten::add", "aten::add_", "aten::sub", "aten::sub_"):
                # alpha!=1 encodes alpha*other +/- self in one aten call.
                alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                amount = nout * (2 if alpha != 1 else 1)
            else:
                amount = nout
        elif name.startswith("aten::_foreach_") and name.endswith(("add_", "sub_", "mul_", "div_")):
            amount = nout
        elif name in _REDUCTIONS:
            source = _first_tensor(args[0]) if args else None
            if source is not None:
                # An N-way reduction needs approximately N-1 adds per result.
                if name in ("aten::aminmax", "custom::aminmax"):
                    amount = 2 * max(0, _numel(source) - nout)
                else:
                    amount = max(0, _numel(source) - nout)
                if "mean" in name:
                    amount += nout  # one normalization divide per result
        elif name in ZERO_COST_ALLOWLIST:
            self.zero_cost[name] += 1
            return out

        if amount is None:
            self.unsupported[name] += 1
        else:
            self._record(self.per_op, name, amount, ins)
            if category == "arithmetic":
                self.arithmetic += int(amount)
            elif category == "nonlinear":
                self.nonlinear += int(amount)
            else:
                self.special += int(amount)
        return out

    @property
    def total(self) -> int:
        return self.arithmetic + self.nonlinear + self.special

    def stats(self) -> dict[str, Any]:
        return {
            "definition": "scalar-op proxy; matrix MAC=2; simple transcendental=1/element; softmax/layernorm/norm/loss use documented formulas",
            "categories": {"arithmetic": self.arithmetic, "nonlinear": self.nonlinear, "special": self.special},
            "total_proxy": self.total,
            "per_op": dict(sorted(self.per_op.items())),
            "unsupported": dict(sorted(self.unsupported.items())),
            "zero_cost_allowlist_hits": dict(sorted(self.zero_cost.items())),
            "zero_cost_allowlist": sorted(ZERO_COST_ALLOWLIST),
            "examples": dict(self.examples),
            "limitations": [
                "Proxy counts are not hardware-exact FLOPs, latency, memory traffic, or energy.",
                "Simple nonlinear functions use one proxy op per element; softmax/layer norm/loss use documented formulas, still not hardware-exact.",
                "Indexing, copies and metadata operations are explicitly excluded from arithmetic totals.",
                "Unknown operators remain visible in unsupported; compare runs only after resolving them.",
            ],
        }

    def stats_json(self, **kwargs) -> str:
        return json.dumps(self.stats(), indent=2, ensure_ascii=False, **kwargs)


def _self_test():
    """Small deterministic accounting check; can run this module as a script."""
    x = torch.ones((2, 3), requires_grad=True)
    w = torch.ones((3, 4), requires_grad=True)
    bias = torch.ones((2, 4), requires_grad=True)
    with ArithmeticCounter() as c:
        y = torch.addmm(bias, x, w)  # 2*2*4*3 + 2*4 bias adds = 56
        loss = y.sum()              # 8 outputs reduced to one = 7 additions
        loss.backward()
    assert c.per_op["aten::addmm"] == 56, c.stats()
    assert c.per_op["aten::sum"] == 7, c.stats()
    assert c.per_op["aten::mm"] == 96, c.stats()  # two matrix-multiply gradients
    assert not c.unsupported, c.stats()

    v = torch.ones((2, 3), requires_grad=True)
    with ArithmeticCounter() as c2:
        v.mean().backward()
    assert c2.per_op["aten::mean"] == 6, c2.stats()
    assert not c2.unsupported, c2.stats()

    q = torch.randn((2, 4), requires_grad=True)
    with ArithmeticCounter() as c3:
        torch.softmax(q, dim=-1).sum().backward()
    assert c3.per_op["aten::_softmax"] == 30, c3.stats()  # 2*(4*4-1)
    assert c3.per_op["aten::_softmax_backward_data"] == 30, c3.stats()
    assert not c3.unsupported, c3.stats()

    q = torch.randn((2, 4), requires_grad=True)
    weight = torch.ones(4, requires_grad=True)
    with ArithmeticCounter() as c4:
        torch.nn.functional.layer_norm(q, (4,), weight).sum().backward()
    assert c4.per_op["aten::native_layer_norm"] == 50, c4.stats()
    assert c4.per_op["aten::native_layer_norm_backward"] == 68, c4.stats()

    logits = torch.randn((2, 3), requires_grad=True)
    target = torch.tensor([1, 2])
    with ArithmeticCounter() as c5:
        torch.nn.functional.nll_loss(logits.log_softmax(dim=1), target).backward()
    assert c5.per_op["aten::nll_loss_forward"] == 4, c5.stats()
    assert c5.per_op["aten::nll_loss_backward"] == 4, c5.stats()
    assert not c5.unsupported, c5.stats()

    q = torch.randn((2, 3), requires_grad=True)
    with ArithmeticCounter() as c6:
        torch.linalg.vector_norm(q, ord=2, dim=1).sum().backward()
    assert c6.per_op["aten::linalg_vector_norm"] == 10, c6.stats()
    assert not c6.unsupported, c6.stats()
    print("compute_budget known-op checks passed")


if __name__ == "__main__":
    _self_test()
