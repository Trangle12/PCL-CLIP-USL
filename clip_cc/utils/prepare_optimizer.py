import torch

def make_vit_optimizer(model):
    """
    Create ViT optimizer.
    
    Params:
        cfg: Config instance.
        model: The model to be optimized.
    Returns:
        An optimizer.
    """
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.00035
        weight_decay = 0.0005
        if "bias" in key:
            lr = 0.00035 * 1.0
            weight_decay = 0.0005
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)

    return optimizer
