from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import re, collections
import regex
from tests.common import gpt2_bytes_to_unicode

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import os
import heapq
import regex
import time
import random
import multiprocessing
from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict, Any, Union
import mmap
import re
from collections import defaultdict

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from tests.hw3.linear import Linear
    
    # 创建 Linear 模块实例
    linear = Linear(
        in_features=d_in,
        out_features=d_out,
        device=weights.device,
        dtype=weights.dtype
    )
    
    # 加载权重（直接赋值，因为权重形状已经是 (d_out, d_in)）
    linear.W.data = weights
    
    # 执行前向传播
    output = linear(in_features)
    
    return output


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    from tests.hw3.embedding import Embedding
    
    # 创建 Embedding 模块实例
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=weights.device,
        dtype=weights.dtype
    )
    
    # 加载权重（直接赋值，因为权重形状已经是 (vocab_size, d_model)）
    embedding.embedding.data = weights
    
    # 执行前向传播
    output = embedding(token_ids)
    
    return output


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    from tests.hw3.swiglu import swiglu

    # 直接调用函数形式的实现，权重已经按 (d_ff, d_model)/(d_model, d_ff) 排列
    return swiglu(
        x=in_features,
        w1=w1_weight,
        w2=w2_weight,
        w3=w3_weight,
    )


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    import math

    # 1) 打分: Q K^T / sqrt(d_k)
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  # (..., queries, keys)

    # 2) 掩码: False 位置置为 -inf，True 保留
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 3) softmax
    attn = torch.softmax(scores, dim=-1)  # (..., queries, keys)

    # 4) 权重求和
    output = torch.matmul(attn, V)  # (..., queries, d_v)

    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    import math
    from tests.adapters import run_scaled_dot_product_attention  # reuse tested SDPA

    # 1) 基本维度
    *batch_dims, seq_len, _ = in_features.shape
    d_k = q_proj_weight.shape[0]  # = d_model // num_heads
    d_v = v_proj_weight.shape[0]  # = d_model // num_heads

    # 2) 一次性计算 Q,K,V 投影（线性，无 bias）
    Q = torch.matmul(in_features, q_proj_weight.transpose(0, 1))  # (..., seq_len, d_k)
    K = torch.matmul(in_features, k_proj_weight.transpose(0, 1))
    V = torch.matmul(in_features, v_proj_weight.transpose(0, 1))  # (..., seq_len, d_v)

    # 3) 重排为多头: (..., num_heads, seq_len, head_dim)
    def split_heads(x, head_dim):
        x = x.view(*batch_dims, seq_len, num_heads, head_dim)
        return x.transpose(-3, -2)  # (..., num_heads, seq_len, head_dim)

    Q = split_heads(Q, d_k // num_heads) if d_k == d_model else split_heads(Q, d_k // num_heads)
    K = split_heads(K, d_k // num_heads) if d_k == d_model else split_heads(K, d_k // num_heads)
    V = split_heads(V, d_v // num_heads) if d_v == d_model else split_heads(V, d_v // num_heads)

    # 这里 d_k = d_v = d_model / num_heads，直接用
    head_dim = d_model // num_heads

    # 4) 构造因果掩码 (seq_len, seq_len) 并广播
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))

    # 5) 计算注意力：把 head 维度当 batch 维度
    attn_out = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    # attn_out: (..., num_heads, seq_len, head_dim)

    # 6) 合并多头: (..., seq_len, num_heads*head_dim) = (..., seq_len, d_model)
    attn_out = attn_out.transpose(-3, -2).contiguous()  # (..., seq_len, num_heads, head_dim)
    attn_out = attn_out.view(*batch_dims, seq_len, d_model)

    # 7) 输出投影
    out = torch.matmul(attn_out, o_proj_weight.transpose(0, 1))  # (..., seq_len, d_model)
    return out


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    import math
    from tests.hw3.rope import RotaryPositionalEmbedding
    from tests.adapters import run_scaled_dot_product_attention  # reuse tested SDPA

    *batch_dims, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads

    # 1) Q,K,V 投影
    Q = torch.matmul(in_features, q_proj_weight.transpose(0, 1))  # (..., seq_len, d_model)
    K = torch.matmul(in_features, k_proj_weight.transpose(0, 1))
    V = torch.matmul(in_features, v_proj_weight.transpose(0, 1))

    # 2) 重排为多头: (..., num_heads, seq_len, head_dim)
    def split_heads(x):
        x = x.view(*batch_dims, seq_len, num_heads, head_dim)
        return x.transpose(-3, -2)  # (..., num_heads, seq_len, head_dim)

    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)

    # 3) 应用 RoPE 到 Q, K（head 维度当 batch 维度）
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)
    rope = RotaryPositionalEmbedding(theta=theta, d_k=head_dim, max_seq_len=max_seq_len, device=in_features.device)
    Q = rope(Q, token_positions)
    K = rope(K, token_positions)

    # 4) 因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))

    # 5) 注意力
    attn_out = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    # (..., num_heads, seq_len, head_dim)

    # 6) 合并头
    attn_out = attn_out.transpose(-3, -2).contiguous()  # (..., seq_len, num_heads, head_dim)
    attn_out = attn_out.view(*batch_dims, seq_len, d_model)

    # 7) 输出投影
    out = torch.matmul(attn_out, o_proj_weight.transpose(0, 1))  # (..., seq_len, d_model)
    return out


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from tests.hw3.rope import RotaryPositionalEmbedding
    
    # 创建 RoPE 模块实例
    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device
    )
    
    # 执行前向传播
    output = rope(in_query_or_key, token_positions)
    
    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    from tests.hw3.transformer_block import TransformerBlock
    
    # 创建 TransformerBlock 实例
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    
    # 将模型移动到输入张量的设备
    device = in_features.device
    block = block.to(device)
    
    # 加载 RMSNorm 的权重
    block.ln1.g.data = weights["ln1.weight"].to(device)
    block.ln2.g.data = weights["ln2.weight"].to(device)
    
    # 提取 attention 和 FFN 的权重
    attn_q_proj_weight = weights["attn.q_proj.weight"].to(device)
    attn_k_proj_weight = weights["attn.k_proj.weight"].to(device)
    attn_v_proj_weight = weights["attn.v_proj.weight"].to(device)
    attn_output_proj_weight = weights["attn.output_proj.weight"].to(device)
    
    ffn_w1_weight = weights["ffn.w1.weight"].to(device)
    ffn_w2_weight = weights["ffn.w2.weight"].to(device)
    ffn_w3_weight = weights["ffn.w3.weight"].to(device)
    
    # 运行前向传播
    with torch.no_grad():
        output = block(
            in_features,
            attn_q_proj_weight=attn_q_proj_weight,
            attn_k_proj_weight=attn_k_proj_weight,
            attn_v_proj_weight=attn_v_proj_weight,
            attn_output_proj_weight=attn_output_proj_weight,
            ffn_w1_weight=ffn_w1_weight,
            ffn_w2_weight=ffn_w2_weight,
            ffn_w3_weight=ffn_w3_weight,
        )
    
    return output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from tests.hw3.rmsnorm import RMSNorm
    
    # 创建 RMSNorm 模块实例
    rmsnorm = RMSNorm(
        d_model=d_model,
        eps=eps,
        device=weights.device,
        dtype=weights.dtype
    )
    
    # 加载权重（直接赋值，因为权重形状已经是 (d_model,)）
    rmsnorm.g.data = weights
    
    # 执行前向传播
    output = rmsnorm(in_features)
    
    return output


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    from tests.hw3.swiglu import silu

    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # raise NotImplementedError
    # 减去维度的最大值，避免exp溢出
    x_max, _ = torch.max(in_features, dim=dim, keepdim=True)
    x_shifted = in_features - x_max

    exp_x = torch.exp(x_shifted)

    exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    out = exp_x / exp_x_sum
    # 测试溢出场景：x + 100，仍应与 expected 相同（减最大值技巧起作用
    return out



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 支持任意批量维度：将所有批维合并为一维
    vocab_size = inputs.shape[-1]
    logits = inputs.reshape(-1, vocab_size)
    tgt = targets.reshape(-1)

    # 1) 数值稳定：减去每个样本的最大 logit
    shifted = logits - logits.max(dim=-1, keepdim=True).values

    # 2) logsumexp 计算分母的 log
    logsumexp = torch.logsumexp(shifted, dim=-1)  # (batch*,)

    # 3) 取出对应目标类的 logit
    target_logit = shifted.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (batch*,)

    # 4) 交叉熵：-log softmax = logsumexp - target_logit
    loss = logsumexp - target_logit  # (batch*,)

    # 5) 返回批次平均
    return loss.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from tests.hw2.tokenizer_encode_decode import Tokenizer
    
    return Tokenizer(vocab, merges, special_tokens)

# ========== 并行预分词辅助函数 ==========
# 全局变量用于多进程
_global_byte_to_unicode = None
_global_PAT = None

def init_worker(byte_to_unicode, PAT):
    """在子进程启动时初始化全局变量，避免每次传递大字典"""
    global _global_byte_to_unicode, _global_PAT
    _global_byte_to_unicode = byte_to_unicode
    _global_PAT = PAT

def pre_tokenize_chunk_worker(chunk_text: str) -> dict:
    """
    工作函数：处理单个文本块
    使用全局变量，避免每次传递大字典
    """
    global _global_byte_to_unicode, _global_PAT
    
    local_vocab = collections.defaultdict(int)
    
    # 对当前 chunk 进行预分词
    words = regex.findall(_global_PAT, chunk_text)
    
    for word in words:
        # 将单词编码为 UTF-8 字节
        word_bytes = word.encode("utf-8")
        # 将每个字节转换为一个单独的bytes对象
        bytes_list = [bytes([x]) for x in word_bytes]
        # 将每个byte映射到对应的Unicode字符
        char_tokens = tuple([_global_byte_to_unicode[b[0]] for b in bytes_list])
        local_vocab[char_tokens] += 1
    
    # 返回普通 dict（multiprocessing 会自动序列化）
    return dict(local_vocab)

def parallel_pre_tokenize(input_path, special_tokens, byte_to_unicode, num_workers=4):
    """
    并行预分词 - 使用参考代码的优化技巧
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 读取整个文件
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # 按 special tokens 分割成 chunks
    if special_tokens:
        chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    else:
        chunks = [text]
    
    # 过滤空 chunk
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    # 如果只有一个 worker 或 chunk 太少，使用串行版本
    if num_workers <= 1 or len(chunks) < num_workers:
        vocab = collections.defaultdict(int)
        for chunk in chunks:
            words = regex.findall(PAT, chunk)
            for word in words:
                word_bytes = word.encode("utf-8")
                bytes_list = [bytes([x]) for x in word_bytes]
                char_tokens = tuple([byte_to_unicode[b[0]] for b in bytes_list])
                vocab[char_tokens] += 1
        return vocab
    
    # 使用进程池并行处理
    with multiprocessing.Pool(
        num_workers,
        initializer=init_worker,  # 子进程启动时调用
        initargs=(byte_to_unicode, PAT)  # 传递给 init_worker 的参数
    ) as pool:
        # 使用 imap 流式处理，chunksize 批量处理减少通信开销
        chunksize = max(1, len(chunks) // (num_workers * 4))  # 动态 chunksize
        chunk_vocabs = list(pool.imap(
            pre_tokenize_chunk_worker, 
            chunks, 
            chunksize=chunksize
        ))
    
    # 合并所有 chunk 的结果
    vocab = collections.defaultdict(int)
    for chunk_vocab in chunk_vocabs:
        for token, freq in chunk_vocab.items():
            vocab[token] += freq
    
    return vocab

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 获取 GPT-2 字节到 Unicode 的映射
    byte_to_unicode = gpt2_bytes_to_unicode()
    # 构建反向映射：Unicode 字符 -> 字节值
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

    def token_str_to_bytes(token_str: str) -> bytes:
        """将 GPT-2 编码的字符串 token 转换为 bytes"""
        try:
            # 尝试使用 GPT-2 映射
            return bytes([unicode_to_byte[char] for char in token_str])
        except KeyError:
            # 如果字符不在映射中，直接使用 UTF-8 编码
            return token_str.encode("utf-8")

    # 根据文件大小决定是否使用并行
    try:
        file_size = os.path.getsize(input_path)
    except OSError:
        file_size = 0
    
    use_parallel = file_size > 1 * 1024 * 1024  # 大于 1MB 才并行
    
    if use_parallel:
        num_workers = min(os.cpu_count() or 1, 4)  # 最多 4 个进程
        vocab = parallel_pre_tokenize(
            input_path=input_path,
            special_tokens=special_tokens,
            byte_to_unicode=byte_to_unicode,
            num_workers=num_workers
        )
    else:
        # 小文件用串行版本（保持原来的逻辑）
        vocab = collections.defaultdict(int)
        try:
            with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            
            if special_tokens:
                chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
            else:
                chunks = [text]
            
            for chunk in chunks:
                words = regex.findall(PAT, chunk)
                for word in words:
                    word_bytes = word.encode("utf-8")
                    bytes_list = [bytes([x]) for x in word_bytes]
                    char_tokens = tuple([byte_to_unicode[b[0]] for b in bytes_list])
                    vocab[char_tokens] += 1
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {input_path}")

    # BPE迭代合并 - 使用增量更新优化
    # 计算需要合并的次数：vocab_size - 初始token数(256字节 + 特殊token)
    initial_vocab_size = 256 + len(special_tokens)
    num_merges = vocab_size - initial_vocab_size
    if num_merges <= 0:
        num_merges = 0
    
    # 初始化 pair_counts（只计算一次）
    pair_counts = collections.defaultdict(int)
    for word, freq in vocab.items():
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            pair_counts[pair] += freq
    
    merges = []
    for i in range(num_merges):
        if not pair_counts:
            break
        
        # 找到最高频的 pair
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates, key=lambda x: (token_str_to_bytes(x[0]), token_str_to_bytes(x[1])))
        
        merges.append(best_pair)
        
        # 增量更新：只更新受影响的 pairs
        pair_merged = best_pair[0] + best_pair[1]
        
        # 找出所有包含 best_pair 的 word，并合并它们
        words_to_update = []
        for word, freq in list(vocab.items()):
            # 检查 word 是否包含 best_pair
            has_pair = False
            for j in range(len(word) - 1):
                if word[j] == best_pair[0] and word[j+1] == best_pair[1]:
                    has_pair = True
                    break
            
            if has_pair:
                words_to_update.append((word, freq))
        
        # 对每个受影响的 word 进行合并
        for word, freq in words_to_update:
            # 1. 先减去旧 pairs 的计数
            for j in range(len(word) - 1):
                old_pair = (word[j], word[j+1])
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            
            # 2. 合并 word
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == best_pair[0] and word[j+1] == best_pair[1]:
                    new_word.append(pair_merged)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            
            # 3. 删除旧的 word，添加新的 word（累加频率）
            del vocab[word]
            new_word_tuple = tuple(new_word)
            vocab[new_word_tuple] = vocab.get(new_word_tuple, 0) + freq
            
            # 4. 加上新 pairs 的计数
            for j in range(len(new_word_tuple) - 1):
                new_pair = (new_word_tuple[j], new_word_tuple[j+1])
                pair_counts[new_pair] += freq
        
        '''
        print(f"Merged {best_pair} in step {i+1}")
        print("vocab: ", vocab)
        print("pairs: ", pairs)
        print("best_pair: ", best_pair)
        print("--------------------------------")
        print("--------------------------------")
        '''

    # 3. 最终结果
    
    # for word, freq in vocab.items():
        # print(f"{word}: {freq}")

    # 构建最终返回格式
    final_vocab = {}
    token_id = 0

    # 1. 添加特殊token
    for special_token in special_tokens:
        final_vocab[token_id] = special_token.encode("utf-8")
        token_id += 1

    # 2. 添加所有单个字节token (0-255)
    for byte_val in range(256):
        if token_id >= vocab_size:
            break
        token_bytes = bytes([byte_val])
        final_vocab[token_id] = token_bytes
        token_id += 1

    # 3. 收集所有出现过的token（从vocab的键中提取）
    all_tokens = set()
    for word in vocab.keys():
        # word现在是tuple
        all_tokens.update(word)

    # 4. 按照merges的顺序添加合并后的token
    seen_merged_tokens = set()
    for token1_str, token2_str in merges:
        if token_id >= vocab_size:
            break
        # 合并后的token
        merged_token_str = token1_str + token2_str
        if merged_token_str not in seen_merged_tokens:
            merged_token_bytes = token_str_to_bytes(merged_token_str)
            if merged_token_bytes not in final_vocab.values():
                final_vocab[token_id] = merged_token_bytes
                token_id += 1
                seen_merged_tokens.add(merged_token_str)

    # 5. 添加其他在vocab中出现但还没添加的token
    for token_str in sorted(all_tokens):
        if token_id >= vocab_size:
            break
        token_bytes = token_str_to_bytes(token_str)
        if token_bytes not in final_vocab.values():
            final_vocab[token_id] = token_bytes
            token_id += 1

    # 转换merges为bytes格式
    merges_bytes = []
    for token1_str, token2_str in merges:
        token1_bytes = token_str_to_bytes(token1_str)
        token2_bytes = token_str_to_bytes(token2_str)
        merges_bytes.append((token1_bytes, token2_bytes))

    return final_vocab, merges_bytes

