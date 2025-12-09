import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        # 存储维度信息和 epsilon
        self.d_model = d_model
        self.eps = eps
        
        # 创建可学习的增益参数 g，形状为 (d_model,)
        # 初始化为全 1，这样初始时 RMSNorm 不会改变输入的尺度
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g)
    
    def forward(self, x):
        """
        前向传播：执行 RMSNorm 归一化
        
        Args:
            x: 输入张量，形状为 (..., d_model)
        
        Returns:
            输出张量，形状与输入相同 (..., d_model)
        """
        # 1. 保存原始数据类型
        in_dtype = x.dtype
        
        # 2. 上转为 float32 以防止平方时溢出
        x = x.to(torch.float32)
        
        # 3. 计算 RMS (Root Mean Square)
        #    对最后一个维度 (d_model) 计算均方根
        #    x 形状: (..., d_model)
        #    squared 形状: (..., d_model)
        squared = x ** 2
        
        #    对最后一个维度求均值，keepdim=True 保持维度以便广播
        #    mean_squared 形状: (..., 1)
        mean_squared = squared.mean(dim=-1, keepdim=True)
        
        #    计算 RMS，加上 eps 防止除零
        #    rms 形状: (..., 1)
        rms = torch.sqrt(mean_squared + self.eps)
        
        # 4. 归一化：将输入除以 RMS
        #    normalized 形状: (..., d_model)
        normalized = x / rms
        
        # 5. 应用增益参数 g（可学习的缩放因子）
        #    self.g 形状: (d_model,)
        #    normalized 形状: (..., d_model)
        #    PyTorch 会自动广播，结果形状: (..., d_model)
        output = normalized * self.g
        
        # 6. 转回原始数据类型
        return output.to(in_dtype)

