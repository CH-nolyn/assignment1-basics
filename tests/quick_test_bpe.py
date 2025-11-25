#!/usr/bin/env python3
"""快速测试 BPE 函数能否运行"""

import sys
from pathlib import Path

# 添加项目路径（修正：文件在 tests/ 目录下，需要向上找根目录）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH

# 使用小文件测试
input_path = FIXTURES_PATH / "corpus.en"

print("开始测试...")
print(f"输入文件: {input_path}")

try:
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    print("\n✓ 函数执行成功！")
    print(f"返回的 vocab 类型: {type(vocab)}")
    print(f"返回的 merges 类型: {type(merges)}")
except Exception as e:
    print(f"\n✗ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()