import torch
import torch.nn.functional as F

from copy import deepcopy
from torch import nn
from modules.lora_pytorch import LoRA

from .routing import get_routing_strategy, TopKRoutingConfig

from typing import (
    cast,
    overload,
)

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module, allows for:
    - Custom routing strategies
    - Use of LoRA experts 
    """
    def __init__(self, module: nn.Module, experts: nn.ModuleList, args):
        super().__init__()
        self.num_experts = len(experts)
        assert self.num_experts > 0, "Select at least 1 expert when using MoE"

        self.base_expert = module # Frozen module on which MoE is built on
        self.experts = experts # List of experts, each is a copy of the module
        
        # Build Gate and Routing strategy
        self.gate = self._build_gate_module(self.base_expert, self.num_experts, type=args.gate_type, bias=args.gate_bias)
        self.route = get_routing_strategy(args.routing_strategy, self._build_routing_config(args.routing_strategy, args))

        self.lora_experts = args.lora_experts


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        gate_logits = self.gate(inputs) # Parse through gate
        weights, selected_experts = self.route(gate_logits) # decide routing
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                    continue  # Skip expert with no assignments
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx]) # expert contribution
        
        if self.lora_experts:
            # If LoRA experts are used, we need compute the output of the main module contribution
            results += self.base_expert(inputs)

        return results
    

    @classmethod
    def _from_linear(cls, module: nn.Linear, num_experts: int, args) -> nn.Module: 
        """Upcycle a linear module into a MoE module with multiple experts."""
        module_copy = deepcopy(module)
        experts = []
        for _ in range(num_experts):
            exp = deepcopy(module_copy) # Copy Everything (weights included, we're upcycling)
            if args.lora_experts:
                # When lora is active, we keep only the LoRA module, discarding
                # the original linear module weights
                exp = LoRA._from_linear(exp, args.lora_experts_rank).lora_module
            experts.append(exp)
        experts = nn.ModuleList(experts)

        return MoE(module, experts, args)


    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        num_experts: int,
        args
    ):
        """Inject MoE recursively into an existing module. (Tested with TransformerEncoder)"""
        if isinstance(module, nn.Linear):
            return MoE._from_linear(module, num_experts, args)
        elif isinstance(module, nn.MultiheadAttention):
            # Attention is unsupported for now
            return module

        for name, child in module.named_children():
            child = cast(nn.Module, child)
            module._modules[name] = cls.from_module(child, num_experts, args)  # type: ignore

        return module
    
    def _build_routing_config(self, routing_strategy, args):
        """Build the routing configuration based on the provided arguments."""
        if routing_strategy == "topk":
            return TopKRoutingConfig(num_experts_per_tok=args.num_experts_per_tok)
        else:
            raise ValueError(f"Unsupported routing strategy: {routing_strategy}")

    def _build_gate_module(self, module: nn.Module, num_experts: int, type: str = "linear", bias: bool = False) -> nn.Module:
        """Build the gate module based on the type specified."""
        if type == "linear":
            assert hasattr(module, 'in_features')
            return nn.Linear(module.in_features, num_experts, bias=bias) # type: ignore
        else:
            raise ValueError(f"Unsupported gate type: {type}")
