import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # 如果包未安装，使用默认版本或从 pyproject.toml 读取
    __version__ = "1.0.6"  # 或者从 pyproject.toml 读取

from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding

__all__ = ["Linear", "Embedding"]