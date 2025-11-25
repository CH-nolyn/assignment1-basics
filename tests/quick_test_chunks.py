#!/usr/bin/env python3
"""快速验证分块功能"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.pretokenization_example import find_chunk_boundaries

# 使用测试数据 - 修复路径（文件在 tests/ 目录下）
fixtures_path = Path(__file__).parent / "fixtures"
test_file = fixtures_path / "tinystories_sample_5M.txt"

print(f"测试文件: {test_file.name}")
print(f"文件大小: {test_file.stat().st_size:,} 字节\n")

with open(test_file, "rb") as f:
    boundaries = find_chunk_boundaries(f, desired_num_chunks=4, split_special_token=b"<|endoftext|>")
    
    print(f"找到 {len(boundaries)} 个边界: {boundaries}\n")
    
    # 显示每个块的信息
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        f.seek(start)
        chunk = f.read(end - start)
        
        # 检查边界是否正确
        starts_ok = chunk.startswith(b"<|endoftext|>") if i > 1 else True
        ends_ok = chunk.endswith(b"<|endoftext|>") if i < len(boundaries) - 1 else True
        
        status = "✓" if (starts_ok and ends_ok) else "✗"
        print(f"{status} 块 {i}: {start:,} - {end:,} ({len(chunk):,} 字节)")