import statistics
from typing import List, Optional, Tuple


def _mean(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _stdev(values: List[Optional[float]]) -> float:
    values = [v for v in values if v is not None]
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0, end - start + 1)
