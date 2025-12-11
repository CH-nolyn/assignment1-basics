import torch
import torch.nn as nn
from tests.hw3.rmsnorm import RMSNorm
from tests.hw3.rope import RotaryPositionalEmbedding
from tests.hw3.swiglu import swiglu
from tests.adapters import run_scaled_dot_product_attention


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Block
    
    结构：
    1. 第一个子层：y = x + MultiHeadSelfAttention(RMSNorm(x))
    2. 第二个子层：y = x + FFN(RMSNorm(x))
    
    其中 FFN 使用 SwiGLU 激活函数。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.head_dim = d_model // num_heads
        
        # RMSNorm 层
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
        # RoPE（用于在 attention 中应用）
        # 注意：device 会在 forward 时根据输入动态设置
        self.rope_theta = theta
        self.rope_max_seq_len = max_seq_len
        
        # Feed-Forward Network (SwiGLU)
        # 注意：权重会在 load_state_dict 时加载，这里不创建 Linear 层
        # 我们会在 forward 中直接使用权重矩阵进行矩阵乘法
        
    def forward(
        self,
        x: torch.Tensor,
        attn_q_proj_weight: torch.Tensor,
        attn_k_proj_weight: torch.Tensor,
        attn_v_proj_weight: torch.Tensor,
        attn_output_proj_weight: torch.Tensor,
        ffn_w1_weight: torch.Tensor,
        ffn_w2_weight: torch.Tensor,
        ffn_w3_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
            attn_q_proj_weight: Q 投影权重，形状为 (d_model, d_model)
            attn_k_proj_weight: K 投影权重，形状为 (d_model, d_model)
            attn_v_proj_weight: V 投影权重，形状为 (d_model, d_model)
            attn_output_proj_weight: 输出投影权重，形状为 (d_model, d_model)
            ffn_w1_weight: FFN W1 权重，形状为 (d_ff, d_model)
            ffn_w2_weight: FFN W2 权重，形状为 (d_model, d_ff)
            ffn_w3_weight: FFN W3 权重，形状为 (d_ff, d_model)
        
        Returns:
            输出张量，形状为 (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # ========== 第一个子层：Multi-Head Self-Attention ==========
        # Pre-norm: RMSNorm(x)
        x_norm = self.ln1(x)  # (batch, seq_len, d_model)
        
        # 计算 Q, K, V
        Q = torch.matmul(x_norm, attn_q_proj_weight.transpose(0, 1))  # (batch, seq_len, d_model)
        K = torch.matmul(x_norm, attn_k_proj_weight.transpose(0, 1))
        V = torch.matmul(x_norm, attn_v_proj_weight.transpose(0, 1))
        
        # 重排为多头: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE 到 Q 和 K
        # token_positions: (batch, seq_len)，每个位置是 [0, 1, 2, ..., seq_len-1]
        # 注意：Q 和 K 的形状是 (batch, num_heads, seq_len, head_dim)
        # RoPE 期望输入形状是 (..., seq_len, d_k)，所以需要将 num_heads 维度视为 batch 维度的一部分
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        # 创建 RoPE 实例（在 forward 时创建以确保设备正确）
        rope = RotaryPositionalEmbedding(
            theta=self.rope_theta,
            d_k=self.head_dim,
            max_seq_len=self.rope_max_seq_len,
            device=x.device
        )
        Q = rope(Q, token_positions)  # (batch, num_heads, seq_len, head_dim)
        K = rope(K, token_positions)
        
        # 因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Scaled Dot-Product Attention
        attn_out = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # attn_out: (batch, num_heads, seq_len, head_dim)
        
        # 合并多头: (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        attn_out = attn_out.view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        attn_out = torch.matmul(attn_out, attn_output_proj_weight.transpose(0, 1))  # (batch, seq_len, d_model)
        
        # 残差连接: x + MHA(RMSNorm(x))
        x = x + attn_out
        
        # ========== 第二个子层：Feed-Forward Network ==========
        # Pre-norm: RMSNorm(x)
        x_norm = self.ln2(x)  # (batch, seq_len, d_model)
        
        # SwiGLU: gate = SiLU(W1 x) ⊙ (W3 x), out = W2 gate
        ffn_out = swiglu(x_norm, ffn_w1_weight, ffn_w2_weight, ffn_w3_weight)
        # ffn_out: (batch, seq_len, d_model)
        
        # 残差连接: x + FFN(RMSNorm(x))
        x = x + ffn_out
        
        return x

