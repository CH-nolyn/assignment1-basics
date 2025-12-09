import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    RoPE 通过旋转矩阵将位置信息编码到查询和键向量中。
    对于位置 i 的向量，应用旋转矩阵 Ri，其中每对维度 (2k-1, 2k) 作为一个 2D 向量旋转。
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 预计算所有位置和所有维度的 cos 和 sin 值
        # 这样可以避免在每次前向传播时重复计算
        
        # 创建位置索引: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # positions 形状: (max_seq_len,)
        
        # 创建频率索引: k ∈ {1, 2, ..., d_k/2}
        # 每个 k 对应一对维度 (2k-2, 2k-1)，即 (0,1), (2,3), (4,5), ...
        # 根据公式: θi,k = i / Θ^((2k-2)/d_k)
        # 所以对于 k=1: 指数 = 0/d_k = 0
        #      k=2: 指数 = 2/d_k
        #      k=3: 指数 = 4/d_k
        #      ...
        # 创建指数: [0, 2/d_k, 4/d_k, ..., (d_k-2)/d_k]
        k_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        # k_indices = [0, 2, 4, ..., d_k-2]，对应 k=1,2,3,... 时的 (2k-2)
        
        # 计算频率: 1 / (theta^((2k-2)/d_k))
        # 对于 k=1: 1/theta^0 = 1
        # 对于 k=2: 1/theta^(2/d_k)
        # 对于 k=3: 1/theta^(4/d_k)
        inv_freqs = 1.0 / (theta ** (k_indices / d_k))
        # inv_freqs 形状: (d_k//2,)
        
        # 计算所有位置和所有频率的角度
        # positions: (max_seq_len, 1)
        # inv_freqs: (1, d_k//2)
        # angles: (max_seq_len, d_k//2)
        positions_expanded = positions.unsqueeze(1)  # (max_seq_len, 1)
        inv_freqs_expanded = inv_freqs.unsqueeze(0)  # (1, d_k//2)
        angles = positions_expanded * inv_freqs_expanded  # (max_seq_len, d_k//2)
        
        # 计算 cos 和 sin
        cos = torch.cos(angles)  # (max_seq_len, d_k//2)
        sin = torch.sin(angles)  # (max_seq_len, d_k//2)
        
        # 注册为 buffer（不参与梯度计算，但会随模型移动设备）
        # persistent=False 表示不会保存到 state_dict
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对输入应用 RoPE 旋转
        
        Args:
            x: 输入张量，形状为 (..., seq_len, d_k)
            token_positions: 位置索引，形状为 (..., seq_len)，每个元素是位置 ID
        
        Returns:
            旋转后的张量，形状与输入相同 (..., seq_len, d_k)
        """
        # x 形状: (..., seq_len, d_k)
        # token_positions 形状: (..., seq_len)
        
        # 1. 根据 token_positions 索引对应的 cos 和 sin 值
        #    self.cos 形状: (max_seq_len, d_k//2)
        #    token_positions 形状: (..., seq_len)
        #    索引后 cos_pos 形状: (..., seq_len, d_k//2)
        cos_pos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin_pos = self.sin[token_positions]  # (..., seq_len, d_k//2)
        
        # 2. 将输入 x 分成奇数和偶数维度
        #    x 形状: (..., seq_len, d_k)
        #    偶数索引: 0, 2, 4, ..., d_k-2
        #    奇数索引: 1, 3, 5, ..., d_k-1
        x_even = x[..., 0::2]  # (..., seq_len, d_k//2) - 偶数维度
        x_odd = x[..., 1::2]    # (..., seq_len, d_k//2) - 奇数维度
        
        # 3. 应用 2D 旋转矩阵
        #    对于每对 (x_even[k], x_odd[k])，应用旋转:
        #    [x_even']   [cos  -sin] [x_even]
        #    [x_odd' ] = [sin   cos] [x_odd ]
        #
        #    即:
        #    x_even' = x_even * cos - x_odd * sin
        #    x_odd'  = x_even * sin + x_odd * cos
        x_even_rot = x_even * cos_pos - x_odd * sin_pos   # (..., seq_len, d_k//2)
        x_odd_rot = x_even * sin_pos + x_odd * cos_pos    # (..., seq_len, d_k//2)
        
        # 4. 将旋转后的奇偶维度重新交错组合
        #    创建输出张量
        output = torch.zeros_like(x)  # (..., seq_len, d_k)
        
        #    将旋转后的值放回原位置
        output[..., 0::2] = x_even_rot  # 偶数位置
        output[..., 1::2] = x_odd_rot   # 奇数位置
        
        return output

