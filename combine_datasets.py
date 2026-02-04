import argparse
import os
import shutil
import glob


def scan_videos(root: str) -> list[str]:
    exts = ("*.MOV", "*.mov", "*.MP4", "*.mp4", "*.AVI", "*.avi", "*.MKV", "*.mkv")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(root, "*", ext)))
    return sorted(set(videos))


def copy_or_link(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Combine two datasets into one folder")
    parser.add_argument("--src_a", required=True, help="first dataset root")
    parser.add_argument("--src_b", required=True, help="second dataset root")
    parser.add_argument("--output_dir", required=True, help="output combined dataset root")
    args = parser.parse_args()

    for src_root in (args.src_a, args.src_b):
        videos = scan_videos(src_root)
        if not videos:
            raise SystemExit(f"No videos found in {src_root}")
        for src in videos:
            label = os.path.basename(os.path.dirname(src))
            dst = os.path.join(args.output_dir, label, os.path.basename(src))
            copy_or_link(src, dst)


if __name__ == "__main__":
    main()
