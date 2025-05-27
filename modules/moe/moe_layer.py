from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from modules.moe.routing import get_routing_strategy
from modules.moe.experts import get_experts

class MoE(nn.Module):

    def __init__(self, args, module: nn.Module, num_experts: int, routing_strategy: str, lora_experts: bool = False):
        super().__init__()
        assert args.moe.num_experts > 0, "Select at least 1 expert when using MoE"

        self.num_experts = num_experts
        self.lora_experts = lora_experts
        self.module = module

        # Simple linear gate for now
        self.gate = nn.Linear(self.module.out_features, num_experts, bias=False)

        if self.lora_experts:
            for param in self.module.parameters():
                param.requires_grad = False

        self.experts = get_experts(args, module, num_experts)
        self.route = get_routing_strategy(args, routing_strategy)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = self.route(gate_logits)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                    continue  # Skip expert with no assignments
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
        
        if self.lora_experts:
            # If LoRA experts are used, we need compute the output of the main module
            results += self.module(inputs)

        return results