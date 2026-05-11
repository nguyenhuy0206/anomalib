"""Convert DeepPCB into Anomalib's Folder layout."""

import shutil
from pathlib import Path

from .config import DATA_ROOT, PACKAGE_ROOT

SRC_ROOT = PACKAGE_ROOT / "DeepPCB-master" / "PCBData"
TARGET_ROOT = DATA_ROOT
TRAINVAL_FILE = SRC_ROOT / "trainval.txt"
TEST_FILE = SRC_ROOT / "test.txt"


def _resolve_paths(relative_path: str) -> tuple[Path, Path]:
    path = SRC_ROOT / relative_path
    base_stem = Path(relative_path).stem
    parent = path.parent

    normal_candidate = parent / f"{base_stem}_temp.jpg"
    abnormal_candidate = parent / f"{base_stem}_test.jpg"

    if normal_candidate.exists() and abnormal_candidate.exists():
        return normal_candidate, abnormal_candidate

    fallback_normal = parent / f"{base_stem}.jpg"
    if fallback_normal.exists() and abnormal_candidate.exists():
        return fallback_normal, abnormal_candidate

    raise FileNotFoundError(
        f"Could not resolve DeepPCB image pair for {relative_path}. "
        f"Checked: {normal_candidate}, {abnormal_candidate}, {fallback_normal}"
    )


def _copy_image(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        return
    shutil.copy2(src, dst)


def prepare_deep_pcb_folder() -> None:
    """Prepare ``pcb_defect/data`` from the DeepPCB source folder."""

    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(f"DeepPCB source folder not found: {SRC_ROOT}")

    train_normal_dir = TARGET_ROOT / "train" / "normal"
    test_normal_dir = TARGET_ROOT / "test" / "normal"
    test_abnormal_dir = TARGET_ROOT / "test" / "abnormal"

    for folder in [train_normal_dir, test_normal_dir, test_abnormal_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if not TRAINVAL_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "DeepPCB split files not found. Make sure trainval.txt and test.txt exist in PCBData."
        )

    with TRAINVAL_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            image_path, _ = line.split()
            normal_image, _ = _resolve_paths(image_path)
            _copy_image(normal_image, train_normal_dir)

    with TEST_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            image_path, _ = line.split()
            normal_image, abnormal_image = _resolve_paths(image_path)
            _copy_image(normal_image, test_normal_dir)
            _copy_image(abnormal_image, test_abnormal_dir)

    print(f"Prepared DeepPCB folder dataset in: {TARGET_ROOT}")
    print(f"  train normal: {train_normal_dir}")
    print(f"  test normal: {test_normal_dir}")
    print(f"  test abnormal: {test_abnormal_dir}")


if __name__ == "__main__":
    prepare_deep_pcb_folder()

