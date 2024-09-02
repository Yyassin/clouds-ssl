import math


def adjust_learning_rate(step, sched_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < sched_config["warmup_steps"]:
        lr = sched_config["max_lr"] * step / sched_config["warmup_steps"]
    else:
        lr = sched_config["min_lr"] + (
            sched_config["max_lr"] - sched_config["min_lr"]
        ) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (step - sched_config["warmup_steps"])
                / (sched_config["total_steps"] - sched_config["warmup_steps"])
            )
        )
    return lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
