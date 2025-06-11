from math import e
import torch
import torch.nn.functional as F

from argparse import Namespace
from dataclasses import dataclass

from copy import deepcopy
from torch import nn
from modules.lora_pytorch import LoRA

from .routing import get_routing_strategy, TopKRoutingConfig

from typing import (
    Optional,
    Tuple,
    cast,
    overload,
)

@dataclass
class MoEOptions:
    num_experts: int = 3
    gate_type: str = "linear"
    gate_bias: bool = False
    routing_strategy: str = "topk"
    num_experts_per_tok: int = 2 # For topk routing
    lora_experts: bool = True
    lora_experts_rank: int = 5

def namespace_to_moe_opt(ns: Namespace) -> Tuple[MoEOptions, Namespace]:
    """Filter Namespace to create MoEOptions and return the unused fields."""
    valid_keys = MoEOptions.__dataclass_fields__.keys()
    filtered_dict = {k: v for k, v in vars(ns).items() if k in valid_keys}
    discarded_dict = {k: v for k, v in vars(ns).items() if k not in valid_keys}
    return MoEOptions(**filtered_dict), Namespace(**discarded_dict)

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module, allows for:
    - Custom routing strategies
    - Use of LoRA experts 
    """
    def __init__(self, module: nn.Module, experts: nn.ModuleList, opt: Optional[MoEOptions] = None):
        super().__init__()
        self.opt = opt if opt is not None else MoEOptions()
        assert len(experts) == self.opt.num_experts, f"Number of experts ({len(experts)}) does not match num_experts option ({self.opt.num_experts})"
        assert self.opt.num_experts > 0, "Select at least 1 expert when using MoE"
        
        self.base_expert = module # Frozen module on which MoE is built on
        self.experts = experts # List of experts, each is a copy of the module
        
        # Build Gate and Routing strategy
        self.gate = self._build_gate_module(self.base_expert, self.opt.num_experts, type=self.opt.gate_type, bias=self.opt.gate_bias)
        self.route = get_routing_strategy(self.opt.routing_strategy, self._build_routing_config(self.opt.routing_strategy, opt))


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
        
        if self.opt.lora_experts:
            # If LoRA experts are used, we need compute the output of the main module contribution
            results += self.base_expert(inputs)

        return results
    

    @classmethod
    def _from_linear(cls, module: nn.Linear, num_experts: int, opt: MoEOptions) -> nn.Module: 
        """Upcycle a linear module into a MoE module with multiple experts."""
        module_copy = deepcopy(module)
        experts = []
        for _ in range(num_experts):
            exp = deepcopy(module_copy) # Copy Everything (weights included, we're upcycling)
            if opt.lora_experts:
                # When lora is active, we keep only the LoRA module, discarding
                # the original linear module weights
                exp = LoRA._from_linear(exp, opt.lora_experts_rank).lora_module
            experts.append(exp)
        experts = nn.ModuleList(experts)
        
        return MoE(module, experts, opt)

    @classmethod
    def from_module(
        cls,
        module: nn.Module, # Module to be converted into MoE
        opt: MoEOptions
    ):
        """Inject MoE recursively into an existing module. (Tested with TransformerEncoder)"""
        if isinstance(module, nn.Linear):
            return MoE._from_linear(module, opt.num_experts, opt)
        elif isinstance(module, nn.MultiheadAttention):
            # Attention is unsupported for now
            return module

        for name, child in module.named_children():
            child = cast(nn.Module, child)
            module._modules[name] = cls.from_module(child, opt.num_experts, opt)  # type: ignore

        return module
    
    def _build_routing_config(self, routing_strategy, opt):
        """Build the routing configuration based on the provided arguments."""
        if routing_strategy == "topk":
            return TopKRoutingConfig(num_experts_per_tok=opt.num_experts_per_tok)
        else:
            raise ValueError(f"Unsupported routing strategy: {routing_strategy}")

    def _build_gate_module(self, module: nn.Module, num_experts: int, type: str = "linear", bias: bool = False) -> nn.Module:
        """Build the gate module based on the type specified."""
        if type == "linear":
            assert hasattr(module, 'in_features')
            return nn.Linear(module.in_features, num_experts, bias=bias) # type: ignore
        else:
            raise ValueError(f"Unsupported gate type: {type}")
