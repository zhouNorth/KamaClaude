# 按整行切分文本并保留重叠行，返回一基行号和原文片段。
def chunk_text(content: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    lines = content.splitlines()
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines) and (size + len(lines[end]) + 1 <= chunk_chars or end == start):
            size += len(lines[end]) + 1
            end += 1
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append((start + 1, end, text))
        if end == len(lines):
            break
        overlap = 0
        next_start = end
        while next_start > start + 1 and overlap < overlap_chars:
            next_start -= 1
            overlap += len(lines[next_start]) + 1
        start = next_start
    return chunks
