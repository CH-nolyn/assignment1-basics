import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力（不含 RoPE）。
    形状约定:
        输入 x: (..., seq_len, d_model)
        输出  : (..., seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 投影矩阵（无 bias，符合作业要求）
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        因果多头自注意力前向传播。
        mask 形状可为 (..., seq_len, seq_len)；为 None 时自动构造下三角因果掩码。
        """
        # x: (..., seq_len, d_model)
        *batch_dims, seq_len, _ = x.shape

        # 1) Q, K, V 投影
        q = self.Wq(x)  # (..., seq_len, d_model)
        k = self.Wk(x)
        v = self.Wv(x)

        # 2) 重排为多头: (..., num_heads, seq_len, d_k)
        def split_heads(t):
            t = t.view(*batch_dims, seq_len, self.num_heads, self.d_k)
            return t.transpose(-3, -2)  # (..., num_heads, seq_len, d_k)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # 3) 注意力分数: QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)  # (..., num_heads, seq_len, seq_len)

        # 4) 因果掩码：仅允许 j <= i
        if mask is None:
            causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal, float("-inf"))
        else:
            # mask 为 True 表示可见；False 表示不可见
            scores = scores.masked_fill(~mask, float("-inf"))

        # 5) softmax
        attn = torch.softmax(scores, dim=-1)  # (..., num_heads, seq_len, seq_len)

        # 6) 加权求和
        context = torch.matmul(attn, v)  # (..., num_heads, seq_len, d_v)

        # 7) 合并多头: (..., seq_len, num_heads * d_v) = (..., seq_len, d_model)
        context = context.transpose(-3, -2).contiguous()  # (..., seq_len, num_heads, d_v)
        context = context.view(*batch_dims, seq_len, self.d_model)

        # 8) 输出投影
        out = self.Wo(context)  # (..., seq_len, d_model)
        return out