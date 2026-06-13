import re
from pathlib import Path

def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Scan output_dir for folders named 'checkpoint-{N}' and return the path
    of the one with the largest N (i.e. the most recent checkpoint).
    Returns None if no checkpoints are found.
    """
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
 
    checkpoints = []
    for p in ckpt_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            checkpoints.append((int(m.group(1)), p))
 
    if not checkpoints:
        return None
 
    latest_step, latest_path = max(checkpoints, key=lambda x: x[0])
    print(f"[Resume] Found checkpoint at step {latest_step}: {latest_path}")
    return str(latest_path)
