from __future__ import annotations


def truncate_for_history(text: str, *, max_chars: int = 1000, head_ratio: float = 0.5) -> str:
    """
    Keep history compact to avoid blowing the model context window.

    Preserve both the start (task framing) and the end (latest state/action cues).
    """
    t = text or ""
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    ratio = float(head_ratio)
    if ratio <= 0.0:
        head = 0
    elif ratio >= 1.0:
        head = max_chars
    else:
        head = int(max_chars * ratio)
    tail = max_chars - head
    return f"{t[:head]}\n…[snip]…\n{t[-tail:]}"


def compact_history_for_prompt(
    history: list[str],
    *,
    keep_last_pairs: int | None = None,
) -> list[str]:
    """Return compact history with first observation and the last N action/observation pairs."""
    if not history:
        return []
    if keep_last_pairs is None or keep_last_pairs <= 0 or len(history) <= 1:
        return list(history)

    rest = history[1:]
    if len(rest) % 2 != 0:
        rest = rest[:-1]
    pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
    tail_pairs = pairs[-keep_last_pairs:] if keep_last_pairs > 0 else []
    out: list[str] = [history[0]]
    for action, obs in tail_pairs:
        out.append(action)
        out.append(obs)
    return out
