import os
from typing import BinaryIO
from pathlib import Path


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage Example
if __name__ == "__main__":
    # 使用相对路径找到测试文件
    # 从 cs336_basics/ 目录到 tests/fixtures/ 的路径
    file_path = Path(__file__).parent.parent / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    
    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        print("请确保在项目根目录运行此脚本")
    else:
        print(f"处理文件: {file_path}")
        print(f"文件大小: {file_path.stat().st_size:,} 字节\n")
        
        with open(file_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            
            print(f"找到 {len(boundaries)} 个边界: {boundaries}\n")
            
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
                f.seek(start)
                chunk = f.read(end - start)
                chunk_text = chunk.decode("utf-8", errors="ignore")
                
                print(f"块 {i}: 位置 {start:,} - {end:,} ({len(chunk):,} 字节)")
                print(f"  开头: {chunk_text[:50].replace(chr(10), '\\n')}...")
                print(f"  结尾: ...{chunk_text[-50:].replace(chr(10), '\\n')}")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                print()