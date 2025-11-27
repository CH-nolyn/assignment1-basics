import json
import time

import pytest

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

def test_train_bpe_data_folder():
    """
    测试 data 文件夹中的数据集
    这个测试不会与参考输出对比，只是验证函数能正常运行并输出正确格式
    """
    from pathlib import Path
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    
    if not data_path.exists():
        pytest.skip(f"数据文件不存在: {data_path}")
    
    vocab, merges = run_train_bpe(
        input_path=data_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    
    # 验证基本格式
    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    assert len(vocab) == 1000
    assert len(merges) == 1000 - 256 - 1  # vocab_size - 256 bytes - 1 special token
    
    # 验证特殊 token
    assert vocab[0] == b"<|endoftext|>"
    
    # 验证字节 tokens
    for byte_val in range(256):
        token_id = 1 + byte_val  # 1 special token + byte_val
        assert token_id in vocab
        assert vocab[token_id] == bytes([byte_val])
    
    # 验证 merges 格式
    for token1, token2 in merges:
        assert isinstance(token1, bytes)
        assert isinstance(token2, bytes)
        assert len(token1) > 0
        assert len(token2) > 0
    
    # 验证特殊 token 不会出现在其他 vocab 值中
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes
    
    print(f"\n✓ 测试通过！")
    print(f"  Vocab 大小: {len(vocab)}")
    print(f"  Merges 数量: {len(merges)}")
    print(f"  前5个 merges: {merges[:5]}")


def test_train_bpe_data_folder_large():
    """
    测试较大的数据集（可选，需要较长时间）
    """
    from pathlib import Path
    import time
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "owt_valid.txt"
    
    if not data_path.exists():
        pytest.skip(f"数据文件不存在: {data_path}")
    
    print(f"\n开始测试大型数据集: {data_path}")
    start_time = time.time()
    
    vocab, merges = run_train_bpe(
        input_path=data_path,
        vocab_size=5000,
        special_tokens=["<|endoftext|>"],
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 基本验证
    assert len(vocab) == 5000
    assert vocab[0] == b"<|endoftext|>"
    
    print(f"\n✓ 大型数据集测试通过！")
    print(f"  训练时间: {elapsed:.2f} 秒")
    print(f"  Vocab 大小: {len(vocab)}")
    print(f"  Merges 数量: {len(merges)}")