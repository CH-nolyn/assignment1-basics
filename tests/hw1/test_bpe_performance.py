#!/usr/bin/env python3

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.adapters import run_train_bpe

# 设置数据文件路径
DATA_PATH = project_root / "data"

# 可用的数据集文件
DATASETS = {
    "tinystories_train": DATA_PATH / "TinyStoriesV2-GPT4-train.txt",
    "tinystories_valid": DATA_PATH / "TinyStoriesV2-GPT4-valid.txt",
    "owt_train": DATA_PATH / "owt_train.txt",
    "owt_valid": DATA_PATH / "owt_valid.txt",
}

def test_bpe_performance(
    dataset_name: str = "tinystories_valid",
    vocab_size: int = 1000,
    special_tokens: list[str] = None,
):
    """
    测试 BPE 训练性能
    
    Args:
        dataset_name: 数据集名称，可选: tinystories_train, tinystories_valid, owt_train, owt_valid
        vocab_size: 词汇表大小
        special_tokens: 特殊token列表
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    # 获取文件路径
    input_path = DATASETS.get(dataset_name)
    if input_path is None:
        print(f"错误: 未知的数据集名称 '{dataset_name}'")
        print(f"可用的数据集: {list(DATASETS.keys())}")
        return
    
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        return
    
    # 显示文件信息
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"文件路径: {input_path}")
    print(f"文件大小: {file_size_mb:.2f} MB")
    print(f"词汇表大小: {vocab_size}")
    print(f"特殊tokens: {special_tokens}")
    print(f"=" * 60)
    
    # 测试训练时间和性能
    print("\n开始训练 BPE...")
    start_time = time.time()
    start_memory = None
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    except ImportError:
        print("提示: 安装 psutil 可以监控内存使用: pip install psutil")
    
    try:
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
        end_time = time.time()
        
        # 计算内存使用
        end_memory = None
        if start_memory is not None:
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = end_memory - start_memory
        
        # 显示结果
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"训练时间: {end_time - start_time:.2f} 秒")
        if start_memory is not None:
            print(f"内存使用: {memory_used:.2f} MB")
        print(f"词汇表大小: {len(vocab)}")
        print(f"合并次数: {len(merges)}")
        print("=" * 60)
        
        # 显示一些统计信息
        print("\n词汇表统计:")
        print(f"  - 特殊tokens数量: {len(special_tokens)}")
        print(f"  - 字节tokens数量: 256")
        print(f"  - 合并tokens数量: {len(merges)}")
        
        # 显示前10个合并
        print("\n前10个合并规则:")
        for i, (token1, token2) in enumerate(merges[:10], 1):
            print(f"  {i}. {token1} + {token2}")
        
        return vocab, merges
        
    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # 示例1: 使用较小的验证集快速测试
    print("测试1: TinyStories 验证集 (较小，适合快速测试)")
    test_bpe_performance(
        dataset_name="tinystories_valid",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"]
    )
    
    print("\n\n")
    
    # 示例2: 使用训练集（较大，需要更长时间）
    # 取消注释以运行
    # print("测试2: TinyStories 训练集 (较大)")
    # test_bpe_performance(
    #     dataset_name="tinystories_train",
    #     vocab_size=5000,
    #     special_tokens=["<|endoftext|>"]
    # )
    
    # 示例3: 使用 OWT 验证集
    # print("测试3: OWT 验证集")
    # test_bpe_performance(
    #     dataset_name="owt_valid",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"]
    # )