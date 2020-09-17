import torch

def alignment(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean().item()


def uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log().item()