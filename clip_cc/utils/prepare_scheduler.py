from .scheduler import WarmupMultiStepLR


def create_scheduler(optimizer):
    scheduler_type ='warmup'
    print("Using warmup scheduler type")

    if scheduler_type == 'warmup':
        lr_scheduler = WarmupMultiStepLR(optimizer, [20, 40], gamma= 0.1,warmup_method='linear',
                                         warmup_factor=0.01,
                                         warmup_iters=10)
    else:
        raise ValueError(f'Invalid scheduler type {scheduler_type}!')

    return lr_scheduler
