from pathlib import Path
import pandas as pd

def _path(cache_dir: str, name: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"{name}.parquet"

def write_parquet(df: pd.DataFrame, cache_dir: str, name: str) -> None:
    _path(cache_dir, name).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_path(cache_dir, name), index=True)

def read_parquet(cache_dir: str, name: str) -> pd.DataFrame | None:
    p = _path(cache_dir, name)
    if not p.exists():
        return None
    return pd.read_parquet(p)
