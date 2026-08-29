import hashlib
import re
from pathlib import Path

from config import DATA_DIR


def file_hash(uploaded_file):
    """Stable identity: same bytes are never indexed twice."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def save_uploaded_file(uploaded_file, paper_id):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(uploaded_file.name).name)
    path = DATA_DIR / f"{paper_id}_{safe_name}"
    if not path.exists():
        path.write_bytes(uploaded_file.getvalue())
    return str(path)
