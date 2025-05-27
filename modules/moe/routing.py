import torch

def get_routing_strategy(args, name: str):
    strategies = {
        "topk": TopKRouting(args),
    }
    if name not in strategies:
        raise ValueError(f"Unsupported routing strategy: {name}")
    return strategies[name]

class TopKRouting:
    def __init__(self, args):
        self.num_experts_per_tok = args.moe.num_experts_per_tok
        assert self.num_experts_per_tok > 0, "num_experts_per_tok must be greater than 0"

    def __call__(self, gate_logits: torch.Tensor, **kwargs):
        return torch.topk(gate_logits, self.num_experts_per_tok)