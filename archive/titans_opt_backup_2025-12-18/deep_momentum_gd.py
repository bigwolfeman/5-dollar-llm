"""
Deep Momentum Gradient Descent (DMGD) with Nested Optimization

Ported from erikl2/nested-learning implementation.
Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
           Behrouz et al., NeurIPS 2025

This optimizer implements the nested learning framework:
- Outer loop: Update model parameters using learned memory
- Inner loop: Train memory modules via internal loss L^(2)

Key features:
- 3-phase step() to prevent OOM (collect -> train memory -> update params)
- SharedMemoryPool with size buckets for efficiency
- Surrogate or L2 regression internal loss modes
- Gradient checkpointing support
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Callable, Any
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class MemoryMLP(nn.Module):
    """
    Learned memory module that processes gradients.

    Takes [gradient, momentum] and outputs a transformed gradient.
    Initialized with small weights for near-identity behavior at start.
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
        use_context: bool = True,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.use_context = use_context

        # Input: gradient (+ momentum if use_context)
        input_dim = param_dim * 2 if use_context else param_dim

        layers = []
        current_dim = input_dim

        for i in range(depth):
            if i == depth - 1:
                # Output layer
                layers.append(nn.Linear(current_dim, param_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                current_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize for near-identity behavior at start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Make output layer even smaller
        last_linear = None
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight, gain=0.01)

    def forward(
        self,
        grad: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process gradient through learned memory."""
        if self.use_context and momentum is not None:
            x = torch.cat([grad, momentum], dim=-1)
        else:
            x = grad
        return self.network(x)


class FactorizedMemoryMLP(nn.Module):
    """
    Low-rank factorized memory for efficiency with large parameter tensors.

    Uses: param_dim -> rank -> core MLP -> rank -> param_dim
    Reduces parameters from O(param_dim * hidden) to O(param_dim * rank + rank * hidden)
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 64,
        rank: int = 16,
        depth: int = 2,
        use_context: bool = True,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.rank = rank
        self.use_context = use_context

        if use_context:
            self.down_proj_grad = nn.Linear(param_dim, rank)
            self.down_proj_mom = nn.Linear(param_dim, rank)
            core_input_dim = rank * 2
        else:
            self.down_proj = nn.Linear(param_dim, rank)
            core_input_dim = rank

        # Core MLP in low-rank space
        core_layers = []
        current_dim = core_input_dim

        for i in range(depth):
            if i == depth - 1:
                core_layers.append(nn.Linear(current_dim, rank))
            else:
                core_layers.append(nn.Linear(current_dim, hidden_dim))
                core_layers.append(nn.LayerNorm(hidden_dim))
                core_layers.append(nn.SiLU())
                current_dim = hidden_dim

        self.core = nn.Sequential(*core_layers)
        self.up_proj = nn.Linear(rank, param_dim)
        self._init_weights()

    def _init_weights(self):
        """Initialize for near-identity behavior at start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)

    def forward(
        self,
        grad: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process gradient through factorized memory."""
        squeeze_output = grad.dim() == 1
        if squeeze_output:
            grad = grad.unsqueeze(0)
            if momentum is not None:
                momentum = momentum.unsqueeze(0)

        if self.use_context and momentum is not None:
            g_low = self.down_proj_grad(grad)
            m_low = self.down_proj_mom(momentum)
            x = torch.cat([g_low, m_low], dim=-1)
        else:
            x = self.down_proj(grad) if hasattr(self, 'down_proj') else self.down_proj_grad(grad)

        out_low = self.core(x)
        output = self.up_proj(out_low)

        if squeeze_output:
            output = output.squeeze(0)
        return output


class SharedMemoryPool(nn.Module):
    """
    Shared memory modules with size buckets for efficiency.

    Parameters are grouped by size into buckets, each with its own memory MLP.
    Supports factorized memory for large buckets.
    """

    def __init__(
        self,
        bucket_sizes: List[int] = [64, 256, 1024, 4096],
        hidden_dim: int = 64,
        depth: int = 2,
        use_factorized: bool = False,
        factorized_rank: int = 16,
        factorize_threshold: int = 1024,
    ):
        super().__init__()
        self.bucket_sizes = sorted(bucket_sizes)
        self.use_factorized = use_factorized
        self.factorize_threshold = factorize_threshold

        self.memories = nn.ModuleDict()
        for size in self.bucket_sizes:
            if use_factorized and size >= factorize_threshold:
                self.memories[str(size)] = FactorizedMemoryMLP(
                    param_dim=size,
                    hidden_dim=hidden_dim,
                    rank=factorized_rank,
                    depth=depth,
                )
            else:
                self.memories[str(size)] = MemoryMLP(
                    param_dim=size,
                    hidden_dim=hidden_dim,
                    depth=depth,
                )

    def get_bucket(self, param_numel: int) -> int:
        """Get appropriate bucket size for a parameter."""
        for size in self.bucket_sizes:
            if param_numel <= size:
                return size
        return self.bucket_sizes[-1]

    def get_memory(self, param_numel: int) -> nn.Module:
        """Get memory module for a parameter size."""
        bucket = self.get_bucket(param_numel)
        return self.memories[str(bucket)]

    def forward(
        self,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        param_numel: int,
    ) -> torch.Tensor:
        """Process gradient through appropriate memory module."""
        bucket = self.get_bucket(param_numel)
        memory = self.memories[str(bucket)]

        if param_numel < bucket:
            # Pad with zeros
            grad_padded = torch.zeros(bucket, device=grad.device, dtype=grad.dtype)
            grad_padded[:param_numel] = grad
            mom_padded = torch.zeros(bucket, device=momentum.device, dtype=momentum.dtype)
            mom_padded[:param_numel] = momentum

            output = memory(grad_padded.unsqueeze(0), mom_padded.unsqueeze(0)).squeeze(0)
            return output[:param_numel]
        else:
            return memory(
                grad[:bucket].unsqueeze(0),
                momentum[:bucket].unsqueeze(0),
            ).squeeze(0)


class DeepMomentumGD(Optimizer):
    """
    Deep Momentum Gradient Descent with nested optimization.

    Implements the full nested learning framework:
    - Memory modules process gradients to produce update directions
    - Memory modules are trained via internal loss (gradient reconstruction)

    Two internal loss modes:
    - 'surrogate': Cosine similarity + magnitude preservation + temporal smoothness
    - 'l2_regression': Paper-exact L² regression (||memory(k) - P @ k||²)

    Args:
        params: Model parameters to optimize
        lr: Learning rate for model parameters
        momentum: Momentum coefficient
        memory_lr: Learning rate for memory module training
        memory_hidden_dim: Hidden dimension of memory MLP
        memory_depth: Depth of memory MLP
        memory_update_freq: How often to update memory (steps)
        bucket_sizes: Bucket sizes for shared memory pool
        weight_decay: Weight decay coefficient
        gradient_checkpointing: Use gradient checkpointing for memory efficiency
        use_factorized_memory: Use factorized memory for large tensors
        factorized_rank: Rank for factorized memory
        internal_loss_mode: 'surrogate' or 'l2_regression'
        l2_projection_lr: Learning rate for L2 projection updates
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        memory_lr: float = 1e-4,
        memory_hidden_dim: int = 64,
        memory_depth: int = 2,
        memory_update_freq: int = 1,
        bucket_sizes: Optional[List[int]] = None,
        weight_decay: float = 0.0,
        gradient_checkpointing: bool = False,
        use_factorized_memory: bool = False,
        factorized_rank: int = 16,
        internal_loss_mode: str = 'surrogate',
        l2_projection_lr: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if internal_loss_mode not in ('surrogate', 'l2_regression'):
            raise ValueError(f"internal_loss_mode must be 'surrogate' or 'l2_regression'")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            memory_lr=memory_lr,
            memory_hidden_dim=memory_hidden_dim,
            memory_depth=memory_depth,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.memory_update_freq = memory_update_freq
        self.step_count = 0
        self.gradient_checkpointing = gradient_checkpointing
        self.internal_loss_mode = internal_loss_mode
        self.l2_projection_lr = l2_projection_lr

        # Memory output history for temporal smoothness
        self._output_history: Dict[int, List[torch.Tensor]] = {}
        self._history_size = 3

        # L2 projection matrices for paper-exact mode
        self._l2_projections: Dict[int, torch.Tensor] = {}

        # Initialize shared memory pool
        if bucket_sizes is None:
            bucket_sizes = [64, 256, 1024, 4096]

        self.shared_memory = SharedMemoryPool(
            bucket_sizes=bucket_sizes,
            hidden_dim=memory_hidden_dim,
            depth=memory_depth,
            use_factorized=use_factorized_memory,
            factorized_rank=factorized_rank,
        )

        self.memory_optimizer = torch.optim.Adam(
            self.shared_memory.parameters(),
            lr=memory_lr,
        )

        # Device tracking
        self._device = None

        # Stats for logging
        self._last_internal_loss = 0.0

    def to(self, device: torch.device) -> 'DeepMomentumGD':
        """Move memory modules to device."""
        self._device = device
        self.shared_memory = self.shared_memory.to(device)
        return self

    def _compute_surrogate_loss(
        self,
        memory_output: torch.Tensor,
        grad: torch.Tensor,
        param_id: int,
    ) -> torch.Tensor:
        """Compute surrogate internal loss (default mode)."""
        # Cosine similarity loss
        if grad.norm() > 1e-8 and memory_output.norm() > 1e-8:
            cosine_sim = F.cosine_similarity(
                memory_output.unsqueeze(0),
                grad.unsqueeze(0),
            )
            reconstruction_loss = 1 - cosine_sim.mean()
        else:
            reconstruction_loss = torch.tensor(0.0, device=grad.device)

        # Magnitude preservation
        target_magnitude = grad.norm()
        output_magnitude = memory_output.norm()
        if target_magnitude > 1e-8:
            magnitude_ratio = output_magnitude / target_magnitude
            magnitude_loss = (magnitude_ratio - 1).pow(2)
        else:
            magnitude_loss = output_magnitude.pow(2)

        # Temporal smoothness
        temporal_loss = torch.tensor(0.0, device=grad.device)
        if param_id in self._output_history and len(self._output_history[param_id]) > 0:
            prev_output = self._output_history[param_id][-1]
            if prev_output.shape == memory_output.shape:
                temporal_loss = (memory_output - prev_output).pow(2).mean() * 0.1

        return reconstruction_loss + 0.1 * magnitude_loss + temporal_loss

    def _compute_l2_regression_loss(
        self,
        memory_output: torch.Tensor,
        grad: torch.Tensor,
        bucket_size: int,
    ) -> torch.Tensor:
        """Compute paper-exact L² regression loss."""
        device = grad.device
        dtype = grad.dtype
        output_dim = memory_output.shape[0]
        input_dim = min(grad.shape[0], output_dim)

        if bucket_size not in self._l2_projections:
            P = torch.eye(output_dim, input_dim, device=device, dtype=dtype) * 0.1
            self._l2_projections[bucket_size] = P

        P = self._l2_projections[bucket_size]
        if P.device != device:
            P = P.to(device)
            self._l2_projections[bucket_size] = P

        grad_for_proj = grad[:input_dim]
        target = P @ grad_for_proj
        loss = (memory_output - target).pow(2).mean()

        # Update projection via delta rule
        with torch.no_grad():
            grad_norm_sq = (grad_for_proj ** 2).sum()
            if grad_norm_sq > 1e-8:
                error = memory_output.detach() - target.detach()
                outer = torch.outer(error, grad_for_proj)
                P_update = self.l2_projection_lr * outer / grad_norm_sq
                self._l2_projections[bucket_size] = P + P_update

        return loss

    def _compute_internal_loss(
        self,
        memory_output: torch.Tensor,
        grad: torch.Tensor,
        param_id: int,
        bucket_size: int,
    ) -> torch.Tensor:
        """Dispatch to appropriate loss mode."""
        if self.internal_loss_mode == 'l2_regression':
            return self._compute_l2_regression_loss(memory_output, grad, bucket_size)
        else:
            return self._compute_surrogate_loss(memory_output, grad, param_id)

    def _update_output_history(self, param_id: int, memory_output: torch.Tensor):
        """Update memory output history for temporal smoothness."""
        if param_id not in self._output_history:
            self._output_history[param_id] = []

        self._output_history[param_id].append(memory_output.detach().clone())

        if len(self._output_history[param_id]) > self._history_size:
            self._output_history[param_id].pop(0)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform optimization step with nested learning.

        3-phase design to prevent OOM:
        1. Collect gradients and compute internal losses
        2. Update memory modules (single backward)
        3. Apply parameter updates
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Phase 1: Collect gradient info and compute internal losses
        internal_losses = []
        param_updates = []

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum_coef = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                param_id = id(p)

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = state['momentum_buffer']

                grad_flat = grad.flatten()
                momentum_flat = momentum_buffer.flatten()
                param_numel = p.numel()

                # Process through memory (detached for outer loop)
                memory_output = self.shared_memory(
                    grad_flat.detach(),
                    momentum_flat.detach(),
                    param_numel,
                )

                # Compute with gradients for internal loss
                if self.gradient_checkpointing:
                    grad_for_ckpt = grad_flat.clone()
                    mom_for_ckpt = momentum_flat.clone()
                    numel = param_numel

                    def make_memory_forward(n):
                        def memory_forward(g, m):
                            return self.shared_memory(g, m, n)
                        return memory_forward

                    memory_output_for_loss = gradient_checkpoint(
                        make_memory_forward(numel),
                        grad_for_ckpt,
                        mom_for_ckpt,
                        use_reentrant=False,
                    )
                else:
                    memory_output_for_loss = self.shared_memory(
                        grad_flat,
                        momentum_flat,
                        param_numel,
                    )

                bucket_size = self.shared_memory.get_bucket(param_numel)
                internal_loss = self._compute_internal_loss(
                    memory_output_for_loss,
                    grad_flat[:memory_output_for_loss.shape[0]],
                    param_id,
                    bucket_size,
                )
                internal_losses.append(internal_loss)

                param_updates.append({
                    'param': p,
                    'state': state,
                    'param_id': param_id,
                    'memory_output': memory_output,
                    'grad': grad,
                    'param_numel': param_numel,
                    'momentum_coef': momentum_coef,
                    'lr': lr,
                })

        # Phase 2: Update memory modules
        if len(internal_losses) > 0 and self.step_count % self.memory_update_freq == 0:
            total_internal_loss = torch.stack(internal_losses).mean()
            self._last_internal_loss = total_internal_loss.item()

            self.memory_optimizer.zero_grad()
            total_internal_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.shared_memory.parameters(),
                max_norm=1.0,
            )
            self.memory_optimizer.step()

        # Phase 3: Apply parameter updates
        for update in param_updates:
            p = update['param']
            state = update['state']
            memory_output = update['memory_output']
            grad = update['grad']
            param_numel = update['param_numel']
            momentum_coef = update['momentum_coef']
            lr = update['lr']
            param_id = update['param_id']

            momentum_buffer = state['momentum_buffer']
            grad_flat = grad.flatten()

            self._update_output_history(param_id, memory_output.detach())

            # Reshape and apply to momentum
            if memory_output.shape[0] < param_numel:
                full_output = grad_flat.clone()
                full_output[:memory_output.shape[0]] = memory_output
                processed_grad = full_output.view_as(grad)
            else:
                processed_grad = memory_output.view_as(grad)

            # Update momentum and parameters
            momentum_buffer.mul_(momentum_coef).add_(processed_grad.detach())
            p.data.add_(momentum_buffer, alpha=-lr)

        self.step_count += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory modules for logging."""
        return {
            'step_count': self.step_count,
            'num_params_tracked': len(self._output_history),
            'internal_loss_mode': self.internal_loss_mode,
            'last_internal_loss': self._last_internal_loss,
            'num_l2_projections': len(self._l2_projections) if self.internal_loss_mode == 'l2_regression' else 0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        # Get base optimizer state
        state = {
            'state': {k: v for k, v in self.state.items()},
            'param_groups': self.param_groups,
            'step_count': self.step_count,
            'shared_memory': self.shared_memory.state_dict(),
            'memory_optimizer': self.memory_optimizer.state_dict(),
            'internal_loss_mode': self.internal_loss_mode,
        }

        # Save L2 projections if using that mode
        if self.internal_loss_mode == 'l2_regression':
            state['l2_projections'] = {k: v.clone() for k, v in self._l2_projections.items()}

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        # Load momentum buffers
        for k, v in state_dict['state'].items():
            self.state[k] = v

        self.step_count = state_dict.get('step_count', 0)
        self.shared_memory.load_state_dict(state_dict['shared_memory'])
        self.memory_optimizer.load_state_dict(state_dict['memory_optimizer'])

        if 'l2_projections' in state_dict:
            self._l2_projections = state_dict['l2_projections']
