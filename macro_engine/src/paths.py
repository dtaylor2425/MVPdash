from pathlib import Path

def project_root() -> Path:
    # src/paths.py -> src -> project root
    return Path(__file__).resolve().parents[1]

def data_path(filename: str) -> Path:
    return project_root() / "data" / filename