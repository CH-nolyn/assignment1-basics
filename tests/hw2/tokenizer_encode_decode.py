
# GPT-2 预分词模式
import regex
from typing import Iterable, Iterator, List

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        # vocab 是：{token_id: bytes}
        # 例如：{0: b' ', 1: b'a', 2: b'hello'}
        # bytes_to_id 是：{bytes: token_id}
        # 例如：{b' ': 0, b'a': 1, b'hello': 2}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # 将 merges 列表转换为字典：{合并对: 优先级索引}
        # 例如：merges = [(b'a', b'b'), (b'c', b'd'), (b'ab', b'c')]
        # 结果：{(b'a', b'b'): 0, (b'c', b'd'): 1, (b'ab', b'c'): 2}
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}

        # 添加特殊token到vocab
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.bytes_to_id:
                self.vocab[next_id] = special_bytes
                self.bytes_to_id[special_bytes] = next_id
                next_id += 1

    def _get_bpe_merges(self, piece: bytes) -> List[bytes]:
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            pairs = [(parts[i], parts[i+1]) for i in range(len(parts)-1)
            if (parts[i], parts[i+1]) in self.merges_priority_map]
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> List[int]:
        # 1: 编码文本为token IDs
        if not text:
            return []
        # 2: 处理特殊 token（按长度降序排序）
        sorted_special = sorted(self.special_tokens, key=len, reverse=True)
        if self.special_tokens:
            pattern = '|'.join(map(regex.escape, sorted_special))
            chunks = regex.split(f'({pattern})', text)
        else:
            chunks = [text]
        # 3:初始化 ids 列表
        ids = []

        # 4: 处理每一个chunk
        for chunk in chunks:
            if not chunk:
                continue
            # 4.1: 处理特殊 token
            if chunk in self.special_tokens:
                special_bytes = chunk.encode("utf-8")
                ids.append(self.bytes_to_id[special_bytes])
            else:
                # 4.2 普通文本， 预分词 + BPE合并
                for word in regex.findall(PAT, chunk):
                    # 应用 BPE 合并
                    merged_pieces = self._get_bpe_merges(word.encode("utf-8"))
                    # 转换为 token_ids
                    for piece in merged_pieces:
                        ids.append(self.bytes_to_id[piece])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码（文件句柄按行迭代，每行是完整的）"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        """解码 token IDs 为文本"""
        all_bytes = b''.join(self.vocab[id] for id in ids if id in self.vocab)
        return all_bytes.decode("utf-8", errors="replace")