import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # 创建 embedding 矩阵，形状为 (num_embeddings, embedding_dim)
        # 使用 nn.Parameter 包装
        # 使用 trunc_normal_ 初始化
        # 处理 device 和 dtype
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.embedding = nn.Parameter(embedding_matrix)

        init.trunc_normal_(self.embedding, mean=0.0, std=0.02, a=-0.04, b=0.04)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids):
        """
        前向传播：根据 token_ids 查找对应的 embedding 向量
        
        Args:
            token_ids: 整数张量，形状为 (...,)，每个元素是 token ID (0 到 num_embeddings-1)
        
        Returns:
            输出张量，形状为 (..., embedding_dim)
        """
        # 使用索引查找：self.embedding[token_ids]
        # token_ids 的形状是 (...,)，输出形状是 (..., embedding_dim)
        return self.embedding[token_ids]

