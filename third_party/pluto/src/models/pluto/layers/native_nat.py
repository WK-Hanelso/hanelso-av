"""Pure-PyTorch, ONNX-exportable reimplementation of NATTEN's
NeighborhoodAttention1D (natten 0.14.6).

Motivation (P2, blocker B1): NATTEN's `NeighborhoodAttention1D` calls a custom
CUDA autograd Function (`natten1dqkrpb` / `natten1dav`) that has NO ONNX
symbolic, so the model cannot be exported. This module reimplements the SAME
math with only standard ops (matmul, softmax, gather, reshape, ...), so it is
ONNX-exportable AND numerically equivalent to NATTEN.

Parameter names / shapes are IDENTICAL to NATTEN's module (`qkv`, `proj`,
`rpb`), so a checkpoint trained with NATTEN loads via `load_state_dict` with no
key remapping and produces the same result.

--- NATTEN 0.14.6 1D semantics (reverse-engineered from the installed source) ---
(natten1d.py + functional.py + csrc/cpu/natten1d*_cpu_kernel.cpp +
 csrc/cpu/natten_cpu_commons.h)

Let k = kernel_size (odd, >1), d = dilation, NS = k // 2 (NEIGHBORHOOD_SIZE),
L = sequence length. For each query position i:

  ni = get_window_start(i, L, k, NS, d)   # first key index of the window
  pi = get_pb_start(i, L, k, NS, d)       # first rpb column for the window

  for ki in 0..k-1:
      key/value position = ki*d + ni
      rpb column         = pi + ki
      attn[i, ki] = (q[i] . k[ki*d+ni]) * scale + rpb[head, pi+ki]

  attn = softmax(attn, dim=-1 over the k neighbors)
  out[i] = sum_ki attn[i, ki] * v[ki*d+ni]

Boundary handling: NATTEN does NOT zero-pad the token set. It keeps a FULL
k-sized window that shifts inward at the edges (get_window_start below). The RPB
index also shifts (get_pb_start) so that the *relative* position of each
neighbor within the shifted window maps to the correct bias column. For an
interior token, pi = NS so the center neighbor (ki==NS) uses rpb column NS (the
"zero relative offset" bias). For an edge token whose window is shifted, pi
compensates so bias columns still track true relative offsets.

get_window_start (dilation<=1):
    max(i - NS, 0) + (i + NS >= L) * (L - i - NS - 1)
get_pb_start (dilation<=1):
    NS + (i < NS)*(NS - i) + (i + NS >= L)*(L - i - 1 - NS)

Dilation>1 formulas (get_window_start / get_pb_start dilation branch) are also
implemented below. planTF uses dilation=1 everywhere (dilations=None), so the
d>1 path is implemented for completeness but is UNTESTED against the oracle.

scale = head_dim ** -0.5, applied to q (matches natten1d.py: q = q * scale).

The forward mirrors natten1d.py exactly, including the pad-when-L<window_size
path (NATTEN right-pads x, computes, then crops back to Lp).
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


def _get_window_start(i: int, L: int, k: int, NS: int, d: int) -> int:
    """Exact port of natten_cpu_commons.h::get_window_start (int arithmetic)."""
    if d <= 1:
        return max(i - NS, 0) + (1 if (i + NS >= L) else 0) * (L - i - NS - 1)
    ni = i - NS * d
    if ni < 0:
        return i % d
    if i + NS * d >= L:
        imodd = i % d
        a = (L // d) * d
        b = L - a
        if imodd < b:
            return L - b + imodd - 2 * NS * d
        return a + imodd - k * d
    return ni


def _get_pb_start(i: int, L: int, k: int, NS: int, d: int) -> int:
    """Exact port of natten_cpu_commons.h::get_pb_start (int arithmetic)."""
    if d <= 1:
        return (
            NS
            + (1 if i < NS else 0) * (NS - i)
            + (1 if (i + NS >= L) else 0) * (L - i - 1 - NS)
        )
    if i - NS * d < 0:
        return k - 1 - (i // d)
    if i + NS * d >= L:
        return (L - i - 1) // d
    return NS


def _build_indices(L: int, k: int, d: int):
    """Precompute, for every query position i and neighbor slot ki, the
    key/value position (ki*d + ni) and the rpb column (pi + ki).

    Returns:
        key_idx: LongTensor [L, k]   key/value position for each (i, ki)
        pb_idx:  LongTensor [L, k]   rpb column for each (i, ki)

    These depend ONLY on static shapes (L, k, d), not on tensor data, so they
    are constants w.r.t. the ONNX trace (no data-dependent control flow).
    """
    NS = k // 2
    key_idx = torch.empty(L, k, dtype=torch.long)
    pb_idx = torch.empty(L, k, dtype=torch.long)
    for i in range(L):
        ni = _get_window_start(i, L, k, NS, d)
        pi = _get_pb_start(i, L, k, NS, d)
        for ki in range(k):
            key_idx[i, ki] = ki * d + ni
            pb_idx[i, ki] = pi + ki
    return key_idx, pb_idx


class NativeNeighborhoodAttention1D(nn.Module):
    """Drop-in, ONNX-exportable replacement for
    natten.NeighborhoodAttention1D (1D neighborhood attention).

    Same __init__ signature and same learnable parameters (`qkv`, `proj`,
    `rpb`) as NATTEN's module, so state_dict is 1:1 compatible.
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1)))
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Read-only index cache keyed by length. We store plain (non-buffer,
        # non-registered) tensors so tracing does NOT see a mutable module-state
        # write in the graph. Building the index tensors from a Python-int L
        # yields a traced CONSTANT (T is fixed per level), which is exactly what
        # we want for ONNX export; it introduces no data-dependent control flow.
        self._idx_cache: dict = {}

    def _indices(self, L: int, device):
        cached = self._idx_cache.get(L)
        if cached is None or cached[0].device != device:
            key_idx, pb_idx = _build_indices(L, self.kernel_size, self.dilation)
            key_idx = key_idx.to(device)
            pb_idx = pb_idx.to(device)
            self._idx_cache[L] = (key_idx, pb_idx)
        return self._idx_cache[L]

    def forward(self, x):
        B, Lp, C = x.shape
        # Force Python ints for the (fixed) length/dims so neighbor-index
        # construction is a compile-time constant in the trace (no data-dep).
        Lp = int(Lp)
        L = Lp
        pad_l = pad_r = 0
        if L < self.window_size:
            pad_r = max(0, self.window_size - L)
            x = pad(x, (0, 0, pad_l, pad_r))
            _, L, _ = x.shape

        # qkv: [B, L, 3, num_heads, head_dim] -> [3, B, num_heads, L, head_dim]
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale  # matches natten1d.py

        key_idx, pb_idx = self._indices(L, x.device)  # [L, k], [L, k]
        kk = self.kernel_size

        # Gather the k neighbor keys/values for every query position.
        # k / v: [B, H, L, hd]; we want [B, H, L, kk, hd] indexed by key_idx[L,kk]
        # index_select over the L axis then reshape -> exact per-(i,ki) mapping.
        flat_idx = key_idx.reshape(-1)  # [L*kk]
        k_neighbors = k.index_select(2, flat_idx).reshape(
            B, self.num_heads, L, kk, self.head_dim
        )
        v_neighbors = v.index_select(2, flat_idx).reshape(
            B, self.num_heads, L, kk, self.head_dim
        )

        # attn[b,h,i,ki] = sum_d q[b,h,i,d] * k_neighbors[b,h,i,ki,d]
        # q: [B,H,L,1,hd] * k_neighbors: [B,H,L,kk,hd] -> sum over hd
        attn = (q.unsqueeze(3) * k_neighbors).sum(-1)  # [B, H, L, kk]

        if self.rpb is not None:
            # rpb: [H, 2k-1]; gather column pb_idx[i,ki] -> [H, L, kk]
            rpb_gathered = self.rpb.index_select(1, pb_idx.reshape(-1)).reshape(
                self.num_heads, L, kk
            )
            attn = attn + rpb_gathered.unsqueeze(0)  # broadcast over B

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # out[b,h,i,d] = sum_ki attn[b,h,i,ki] * v_neighbors[b,h,i,ki,d]
        out = (attn.unsqueeze(-1) * v_neighbors).sum(3)  # [B, H, L, hd]

        out = out.permute(0, 2, 1, 3).reshape(B, L, C)
        if pad_r:
            out = out[:, :Lp, :]

        return self.proj_drop(self.proj(out))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )
