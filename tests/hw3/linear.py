import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 创建参数 W，形状为 (out_features, in_features)
        # 使用 nn.Parameter 包装
        # 使用 trunc_normal_ 初始化
        # 处理 device 和 dtype
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.W = nn.Parameter(W)

        init.trunc_normal_(self.W, mean=0.0, std=0.02, a=-0.04, b=0.04)

        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        """
        前向传播：执行线性变换 y = x @ W.T
        
        Args:
            x: 输入张量，形状为 (..., in_features)
        
        Returns:
            输出张量，形状为 (..., out_features)
        """
        return x @ self.W.T
