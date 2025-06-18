"""DeMo: Decoupled Momentum Optimization

This implements the DeMo fused optimizer and data parallel algorithm.
It is recommended to use DeMo as the base data parallelism.
In an exisiting codebase that uses PyTorch DDP, wrap your forward-backward in 
`torch.distributed.DistributedDataParallel.no_sync` to disable external gradient synchronization.
See https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
"""

import math
import torch
import torch.fft
import torch.distributed as dist

from einops import rearrange
from typing import Optional, Callable

import tplr

class DeMo(torch.optim.SGD):
    def __init__(
        self,
        params,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        use_sign: bool = False,
        grad_val_multiplier: float = 1.0,
        use_grad_normalization: bool = False,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        process_group: Optional[dist.ProcessGroup] = None,
        safety_grad_clip_min: float = -100.0,  
        safety_grad_clip_max: float = 100.0,  
        **kwargs,
    ):
        super().__init__(
            params,
            foreach=False,
            momentum=momentum,
            dampening=0.0,
            nesterov=nesterov,
            maximize=False,
            weight_decay=0.0,
            **kwargs,
        )
        if safety_grad_clip_max < 100: 
            use_grad_normalization = False
            tplr.logger.warning(
                f"Disabling gradient normalization because safety_grad_clip_max ({safety_grad_clip_max}) < 800. "
                "Safety clipping and gradient normalization cannot be used together."
            )
        
        self.compression_decay = compression_decay
        self.compression_chunk = compression_chunk
        self.compression_topk = compression_topk
        self.process_group = process_group
        self.weight_decay = weight_decay
        self.use_sign = use_sign
        self.use_grad_normalization = use_grad_normalization
        self.use_quantization = use_quantization
        self.grad_val_multiplier = grad_val_multiplier
        self.safety_grad_clip_min = safety_grad_clip_min
        self.safety_grad_clip_max = safety_grad_clip_max

        if self.compression_topk <= 0:
            raise ValueError("topk_size has to be positive")
        if self.compression_chunk <= 0:
            raise ValueError("chunk_size has to be positive")
        if self.compression_decay < 0:
            raise ValueError("Negative compression_decay is currently not supported")
        if self.compression_decay >= 1:
            raise ValueError("Values of compression_decay bigger or equal to 1.0 is currently not supported")

        self.demo_state = {}
        self._init_demo_states()
        self._init_opt_parameters()

        self.default_dtype = self._find_dtype()
        self.transform = TransformDCT(self.param_groups, self.compression_chunk)
        self.compress = CompressDCT(use_quantization, quantization_bins, quantization_range)

    def _find_dtype(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    return p.dtype
        return torch.float32

    def _init_demo_states(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.demo_state[p] = {}

    def _state_parameter(self, p):
        if p not in self.demo_state:
            self.demo_state[p] = {}
        return self.demo_state[p]

    def _init_opt_parameters(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self._state_parameter(p)

                    state["step"] = 0
                    state["delta"] = torch.zeros_like(p)

    def _demo_all_gather(self, sparse_idx, sparse_val):
        world_size = dist.get_world_size() if self.process_group is None else self.process_group.size()

        # Gather all the idx and vals
        sparse_idx_list = [torch.zeros_like(sparse_idx) for wi in range(world_size)]
        sparse_val_list = [torch.zeros_like(sparse_val) for wi in range(world_size)]

        sparse_idx_handle = dist.all_gather(sparse_idx_list, sparse_idx, group=self.process_group, async_op=True)
        sparse_val_handle = dist.all_gather(sparse_val_list, sparse_val, group=self.process_group, async_op=True)

        sparse_idx_handle.wait()
        sparse_val_handle.wait()

        return sparse_idx_list, sparse_val_list


    @torch.no_grad()
    def step(self, closure: Callable | None = None):

        self.data_transmit = 0
        self.data_receive = 0

        _logged_clamp = False
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._state_parameter(p)

                # Update step
                state["step"] += 1

                # Step-Weight decay
                if self.weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * self.weight_decay)

                # Decay delta
                if self.compression_decay != 1:
                    state["delta"].mul_(self.compression_decay)

                # Add delta to new gradient
                state["delta"].add_(p.grad, alpha=lr)

                # Compress delta
                sparse_idx, sparse_val, xshape, totalk, quant_params_local = self.compress.compress(
                    self.transform.encode(state["delta"]), self.compression_topk
                )
                sparse_val *= self.grad_val_multiplier

                # Estimate transmitted delta
                transmit_grad = self.transform.decode(
                    self.compress.decompress(p, sparse_idx, sparse_val, xshape, totalk, quant_params_local)
                )

                # Remove transmitted from delta
                state["delta"].sub_(transmit_grad)

                # All-gather
                sparse_idx_gather, sparse_val_gather = self._demo_all_gather(sparse_idx, sparse_val)

                # Log I/O data size
                self.data_transmit += sparse_idx.nbytes + sparse_val.nbytes
                for si, v in zip(sparse_idx_gather, sparse_val_gather):
                    self.data_receive += si.nbytes + v.nbytes

                # Calculate worker norms and derive clipping threshold
                worker_norms = torch.stack([torch.norm(sparse_vals, dim=1, p=2) for sparse_vals in sparse_val_gather], dim=1)
                median_norm = torch.median(worker_norms)
                
                # Clamp median_norm between safety bounds to prevent anomalous workers
                clip_thresh = torch.clamp(
                    median_norm,
                    min=self.safety_grad_clip_min,
                    max=self.safety_grad_clip_max
                )
                if not _logged_clamp and clip_thresh != median_norm and clip_thresh.device.index == 0:
                    _logged_clamp = True # avoid spams
                    tplr.logger.warning(
                        f"Median norm of gradients across workers: {median_norm:.2f}, "
                        f"Clipping threshold set to: {clip_thresh:.2f}."
                    )

                # Decode grad from all nodes with clipping
                new_grad = self.transform.decode(
                    self.compress.batch_decompress(
                        p, 
                        sparse_idx_gather, 
                        sparse_val_gather, 
                        xshape, 
                        totalk, 
                        quant_params_local,
                        normalise=self.use_grad_normalization,  # Disable old normalization
                        clip_norm_val=clip_thresh  # Use clipping threshold
                    )
                )

                # Set grad to values
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)

                # Sign-SGD
                if self.use_sign:
                    p.grad.sign_()

        # SGD step
        return super().step(closure)
    
class TransformDCT:
    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm="ortho"):
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    # Get the closest smallest divisor to the targeted DCT size
                    sc = _get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc

                    # Pregenerate DCT basis matrices
                    if sc not in self.f_dict:
                        I = torch.eye(sc)
                        self.f_dict[sc] = _dct(I, norm=norm).to(p.dtype).to(p.device)
                        self.b_dict[sc] = _idct(I, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x):
        if len(x.shape) > 2:  # 2D weights
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w

            x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")

        else:  # 1D weights
            n1 = x.shape[1]
            n1w = self.b_dict[n1].to(x.device)
            self.b_dict[n1] = n1w

            x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class CompressDCT:
    @torch.no_grad()
    def __init__(
        self,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ):
        self.use_quantization = use_quantization
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = (
                quantization_range  # Quantization range in standard deviations
            )

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @torch.no_grad()
    def compress(self, x, topk):
        xshape = x.shape
        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        idx_int64 = torch.topk(
            x.abs(), k=topk, dim=-1, largest=True, sorted=False
        ).indices
        val = torch.gather(x, dim=-1, index=idx_int64)

        # Apply 8-bit quantization if enabled
        if self.use_quantization:
            idx = idx_int64.to(torch.int16)
            idx = idx.to(torch.int64)  # Ensure idx is int64 for NCCL errors
            val, quant_params = self._quantize_values(val)
        else:
            idx = idx_int64
            quant_params = None

        return idx, val, xshape, totalk, quant_params

    @torch.no_grad()
    def decompress(self, p, idx, val, xshape, totalk, quantize_params=None):
        # Dequantize if values were quantized
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Cast back to int64 before using scatter/gather
        idx_int64 = idx.to(torch.int64)
        x.scatter_reduce_(
            dim=-1, index=idx_int64, src=val, reduce="mean", include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x (h w) -> y x h w", h=xshape[2])

        return x

    @torch.no_grad()
    def batch_decompress(
        self, p, idx, val, xshape, totalk, quantize_params=None, normalise=True, clip_norm_val=None
    ):
        """
        Decompress multiple tensors in batch mode with optional gradient norm clipping.
        """
        # Ensure idx and val are lists
        if not isinstance(idx, list):
            idx = [idx]
        if not isinstance(val, list):
            val = [val]

        # Handle quantization parameters
        if quantize_params is not None:
            if not isinstance(quantize_params, list):
                quantize_params = [quantize_params] * len(val)

        # Process values - dequantize and apply clipping if needed
        processed_vals = []
        for i in range(len(val)):
            v = val[i].to(p.device)

            # Dequantize if we have quantization parameters
            if self.use_quantization and quantize_params and i < len(quantize_params):
                v = self._dequantize_values(v, quantize_params[i])


            # Apply L2 normalization to this individual tensor's values
            if normalise:
                eps = 1e-8
                if len(v.shape) == 3:  # 2D weights
                    l2_norm = torch.norm(v, p=2, dim=2, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 2:  # 1D weights (biases)
                    l2_norm = torch.norm(v, p=2, dim=1, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 1:  # Single values
                    l2_norm = torch.norm(v, p=2)
                    if l2_norm > eps:
                        v = v / l2_norm
            # Apply gradient norm clipping if threshold provided
            elif clip_norm_val is not None:
                current_norm = torch.norm(v.float())
                if current_norm > clip_norm_val:
                    clip_factor = clip_norm_val / current_norm
                    v = v * clip_factor

            processed_vals.append(v)

        # Concatenate everything
        idx_concat = torch.cat([i.to(p.device) for i in idx], dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        # Use decompress without quantization (since we already dequantized)
        return self.decompress(
            p, idx_concat, val_concat, xshape, totalk, quantize_params=None
        )

    @torch.no_grad()
    def _quantize_values(self, val):
        """
        Quantize values to 8-bit representation with statistical approach

        Args:
            val: Tensor of values to quantize

        Returns:
            tuple: (quantized_values, quantization_parameters)
        """
        # Statistical quantization approach
        offset = self.n_bins // 2  # 128 for 8-bit
        shift = val.mean()

        # Center tensor around mean
        centered_val = val - shift

        # Calculate standard deviation (unbiased)
        std_unbiased = centered_val.norm() / math.sqrt(centered_val.numel() - 1)

        # Compute scale factor based on standard deviation range
        scale = self.range_in_sigmas * std_unbiased / self.n_bins

        # Ensure scale is not zero to avoid NaN
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = 1.0

        # Quantize to 8-bit representation
        centered_val = centered_val.to(torch.float32)
        quantized_val = (
            (centered_val / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        # Create lookup table by computing mean values for each bucket
        device = quantized_val.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, quantized_val.flatten().long(), centered_val.flatten())
        counts.scatter_add_(
            0, quantized_val.flatten().long(), torch.ones_like(centered_val.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))

        # Store quantization parameters for dequantization
        orig_dtype = val.dtype
        quant_params = (shift, scale, offset, lookup, orig_dtype)

        return quantized_val, quant_params

    @torch.no_grad()
    def _dequantize_values(self, val, quant_params):
        """
        Dequantize 8-bit values back to original representation

        Args:
            val: Quantized uint8 tensor
            quant_params: Tuple of (shift, scale, offset, lookup, orig_dtype)

        Returns:
            Dequantized tensor in original dtype
        """
        if quant_params is None:
            return val

        shift, scale, offset, lookup, orig_dtype = quant_params

        # Ensure lookup is on the same device as val
        if isinstance(lookup, torch.Tensor):
            lookup = lookup.to(val.device)

        # Convert quantized values back using lookup table
        dequantized = lookup[val.long()]

        # Apply scale and shift to get back original distribution
        val = dequantized + shift
        val = val.to(orig_dtype)

        return val


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n):
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n):
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        # Find all the products that are divisors of n
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n