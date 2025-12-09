import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU/Swish 激活: SiLU(x) = x * sigmoid(x)
    支持任意形状输入，逐元素计算。
    """
    return x * torch.sigmoid(x)


def swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU 前馈计算:
        gate = SiLU(W1 x) ⊙ (W3 x)
        out  = W2 gate

    形状约定:
        x   : (..., d_model)
        w1  : (d_ff, d_model)
        w2  : (d_model, d_ff)
        w3  : (d_ff, d_model)
        out : (..., d_model)
    """
    # 上投 (批维保持不变)
    a = x @ w1.T  # (..., d_ff)
    b = x @ w3.T  # (..., d_ff)

    # 门控
    gated = silu(a) * b  # (..., d_ff)

    # 下投回 d_model
    out = gated @ w2.T  # (..., d_model)
    return out

