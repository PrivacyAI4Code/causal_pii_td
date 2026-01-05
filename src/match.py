import ahocorasick
from typing import List, Tuple, Optional, Dict, Any
import unicodedata
from multiprocessing import Pool, cpu_count

def _normalize(s: str, ignore_case: bool) -> str:
    # 统一做NFKC归一化，减少全角/变体影响；可按需关掉
    s = unicodedata.normalize("NFKC", s)
    return s.lower() if ignore_case else s

def build_automaton(patterns: List[str], ignore_case: bool = False) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()
    seen = set()
    for p in patterns:
        if not p:
            continue
        q = _normalize(p, ignore_case)
        if q in seen:
            continue
        seen.add(q)
        A.add_word(q, q)  # value = 命中串本身
    A.make_automaton()
    return A

def match_first(text: str, A: ahocorasick.Automaton, ignore_case: bool=False) -> Optional[Tuple[int,int,str]]:
    s = _normalize(text, ignore_case)
    for end_idx, val in A.iter(s):
        start_idx = end_idx - len(val) + 1
        return (start_idx, end_idx, val)
    return None

def match_all(text: str, A: ahocorasick.Automaton, ignore_case: bool=False) -> List[Tuple[int,int,str]]:
    s = _normalize(text, ignore_case)
    hits = []
    for end_idx, val in A.iter(s):
        start_idx = end_idx - len(val) + 1
        hits.append((start_idx, end_idx, val))
    return hits

def _worker(args):
    """Worker function for multiprocessing - must be at module level for pickling"""
    sub_texts, pats, ig, ret_all = args
    A_local = build_automaton(pats, ig)
    out = []
    if ret_all:
        for t in sub_texts:
            out.append(match_all(t, A_local, ig))
    else:
        for t in sub_texts:
            out.append(match_first(t, A_local, ig))
    return out

def batch_match(
    texts: List[str],
    patterns: List[str],
    ignore_case: bool = False,
    return_all: bool = False,
    parallel: bool = False,
    workers: Optional[int] = None,
) -> List[Optional[Any]]:
    """
    texts: 你的全部 middle 文本
    patterns: 1000+ 模式串
    return_all=False -> 每条文本返回首个命中 or None
    return_all=True  -> 返回所有命中列表(可能为空列表)
    parallel=True    -> 多进程分块（对超大批量文本更有用）
    """
    if not texts:
        return []

    if parallel:
        # 为了避免在进程间传递 Automaton（不总是可靠），
        # 这里选择“每个进程各自构建一个 A”。
        # 对 1000+ 模式，构建开销通常也很小（毫秒到几百毫秒级）。
        if workers is None:
            workers = max(1, min(cpu_count(), 8))  # 适度上限
        chunk = max(1, len(texts) // (workers * 4))


        tasks = []
        for i in range(0, len(texts), chunk):
            tasks.append((texts[i:i+chunk], patterns, ignore_case, return_all))

        with Pool(processes=workers) as pool:
            parts = pool.map(_worker, tasks)
        # 拼接
        result = []
        for part in parts:
            result.extend(part)
        return result

    else:
        # 单进程：建一次 A，扫所有文本（通常已经很快）
        A = build_automaton(patterns, ignore_case)
        if return_all:
            return [match_all(t, A, ignore_case) for t in texts]
        else:
            return [match_first(t, A, ignore_case) for t in texts]
