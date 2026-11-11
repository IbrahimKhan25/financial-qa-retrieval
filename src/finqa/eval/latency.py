"""Compare the latency of two versions of the code measured on the same machine in the same job."""
from statistics import median


def compare(base_p95: list[float], head_p95: list[float], max_ratio: float) -> dict:
    """Each list holds one p95 per round. The median over rounds ignores a single noisy round.

    The gate is a ratio because CI machines differ in speed from run to run, so a fixed millisecond limit would
    either fail on a slow machine or be too loose to catch anything on a fast one.
    """
    if not base_p95 or not head_p95:
        raise ValueError("need at least one round for each version")
    base, head = median(base_p95), median(head_p95)
    if base <= 0:
        raise ValueError("baseline p95 must be positive")
    ratio = head / base
    return {"base_p95_ms": base, "head_p95_ms": head, "ratio": ratio, "max_ratio": max_ratio,
            "passed": ratio <= max_ratio}
