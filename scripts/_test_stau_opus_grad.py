import torch
import torch.nn as nn
from stau_opus_operator import STauOpusAttentionFunction


def test_numerical_grad():
    torch.manual_seed(0)
    B, H, L, D = 1, 2, 8, 16
    Q = torch.randn(B, H, L, D, dtype=torch.float64, requires_grad=True)
    K = torch.randn(B, H, L, D, dtype=torch.float64, requires_grad=True)
    V = torch.randn(B, H, L, D, dtype=torch.float64, requires_grad=True)
    log_tau = nn.Parameter(torch.zeros(H, dtype=torch.float64))
    log_alpha = nn.Parameter(torch.zeros(H, dtype=torch.float64))

    def fn(Q, K, V, log_tau, log_alpha):
        out, _ = STauOpusAttentionFunction.apply(Q, K, V, log_tau, log_alpha, "softplus")
        return out

    ok = torch.autograd.gradcheck(fn, (Q, K, V, log_tau, log_alpha), eps=1e-6, atol=1e-4, rtol=1e-3)
    print("gradcheck:", ok)
    return ok


if __name__ == "__main__":
    test_numerical_grad()
