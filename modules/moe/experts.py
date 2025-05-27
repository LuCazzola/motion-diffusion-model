from copy import deepcopy
from typing import List
from torch import nn
from modules.lora_pytorch import LoRA

def get_experts(args, module: nn.Module, num_experts: int) -> nn.ModuleList:
    
    module_copy = deepcopy(module)
    
    # unfreeze module copied parameters if LoRA experts are used
    # As they should be trainable, unlike the original module
    if args.moe.lora_experts:
        for param in module_copy.parameters():
            param.requires_grad = True

    experts = []
    for _ in range(num_experts):
        expert = deepcopy(module_copy) # Copy Everything (weights included, we're upcycling)
        if args.moe.lora_experts:
            expert = LoRA.from_module(expert, rank=args.lora.rank, lora_ff=True).lora_module
        experts.append(expert)

    return nn.ModuleList(experts)
