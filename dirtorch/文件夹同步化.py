from pathlib import Path
import shutil

SOURCE_ROOT = Path(r"C:\Users\surface\Desktop\jpg-raw\AgeDB\jpg_raw")
DEST_ROOT = Path(r"C:\Users\surface\Desktop\output")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def copy_missing_images() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_ROOT}")
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_file in SOURCE_ROOT.rglob("*"):
        if not src_file.is_file():
            continue
        if src_file.suffix.lower() not in IMAGE_EXTS:
            continue

        rel_path = src_file.relative_to(SOURCE_ROOT)
        dest_file = DEST_ROOT / rel_path
        if dest_file.exists():
            continue

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied += 1

    print(f"Copied {copied} new image(s).")


if __name__ == "__main__":
    copy_missing_images()