#!/usr/bin/env python3
"""
ColonoMind — Full Experiment Results & Plots Archiver to Rclone Remote (clara_drive)
====================================================================================
This script compresses all experiment results, figures, ROC curves, confusion matrices,
metrics JSON, logs, model weights, and artifacts into a dated `.tar.gz` archive,
then transfers the backup directly to an rclone remote (default: clara_drive:).

Features:
- Auto-detects Result directories on DGX server and local setups.
- Supports filtering: include all artifacts, or optionally exclude raw .h5 weights if user wants lightweight tar.gz.
- Uses pigz (parallel gzip) if available for super fast compression of large result pools.
- Verifies archive integrity with `tar -tzf`.
- Directly executes `rclone copy` or `rclone move` with progress bar (--progress).
- Resilient retry logic for flaky network transfers.
"""

import os
import sys
import time
import shutil
import tarfile
import argparse
import subprocess
from datetime import datetime

# Common result locations on DGX and workspace
DEFAULT_CANDIDATE_DIRS = [
    # DGX server locations
    "../Result",
    "./Result",
    "/home/D13K48009/raid/Clara/colonomind-train/Result",
    "/home/D13K48009/raid/Clara/Result",
    "/raid/D13K48009/Clara/colonomind-train/Result",
    "/raid/D13K48009/Clara/Result",
    # Local & Manuscript results
    "./Manuscript_Consistent_Results",
    "./Manuscript_Mock_Results",
    "./Archive_Results",
    "./Dummy_ROC_Images",
    "./NewFallbackFigure"
]

def run_command(cmd, shell=True, check=True):
    """Run shell command with streaming output."""
    print(f"\n▶ Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    res = subprocess.run(cmd, shell=shell, check=check, text=True)
    return res.returncode

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def check_rclone_remote(remote_name):
    """Check if the specified rclone remote exists in config."""
    clean_name = remote_name.rstrip(":")
    try:
        proc = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, check=True)
        remotes = [r.rstrip(":") for r in proc.stdout.strip().split("\n") if r]
        if clean_name in remotes:
            print(f"✅ Rclone remote '{remote_name}' found in configuration.")
            return True
        else:
            print(f"⚠️ Warning: Remote '{clean_name}' not found in `rclone listremotes`.")
            print(f"   Available remotes: {remotes}")
            return False
    except FileNotFoundError:
        print("❌ Error: `rclone` binary not found in PATH. Please ensure rclone is installed.")
        return False
    except Exception as e:
        print(f"⚠️ Warning when checking rclone: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Archive all ColonoMind results & upload to clara_drive")
    parser.add_argument("--src-dirs", nargs="+", default=None,
                        help="Specific directories to include in archive (default: auto-detected Result folders)")
    parser.add_argument("--output-dir", default="./backups",
                        help="Directory where local .tar.gz will be created before transfer (default: ./backups)")
    parser.add_argument("--rclone-remote", default="clara_drive:ColonoMind_Backups",
                        help="Rclone remote destination (default: clara_drive:ColonoMind_Backups)")
    parser.add_argument("--name-tag", default="colonomind_all_results_backup",
                        help="Prefix name for the tar.gz file")
    parser.add_argument("--exclude-h5", action="store_true",
                        help="Exclude large .h5 weights and only backup plots, figures, JSONs, logs, and scalers")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Only create tar.gz archive locally, do not upload via rclone")
    parser.add_argument("--keep-local", action="store_true", default=True,
                        help="Keep the local tar.gz after successful rclone transfer")
    args = parser.parse_args()

    print("=" * 80)
    print("📦 COLONOMIND EXPERIMENT RESULTS & FIGURES BACKUP UTILITY")
    print("=" * 80)

    # 1. Identify existing source directories
    resolved_dirs = []
    if args.src_dirs:
        for p in args.src_dirs:
            ap = os.path.abspath(os.path.expanduser(p))
            if os.path.exists(ap):
                resolved_dirs.append(ap)
            else:
                print(f"⚠️ Warning: User specified directory not found: {p}")
    else:
        for p in DEFAULT_CANDIDATE_DIRS:
            ap = os.path.abspath(os.path.expanduser(p))
            if os.path.exists(ap) and ap not in resolved_dirs:
                resolved_dirs.append(ap)

    if not resolved_dirs:
        print("❌ Error: No valid Result or Experiment directories found to backup!")
        print("   Searched locations:")
        for c in DEFAULT_CANDIDATE_DIRS:
            print(f"   - {c}")
        sys.exit(1)

    print(f"\n📂 Discovered {len(resolved_dirs)} Target Directories for Backup:")
    total_raw_size = 0
    for d in resolved_dirs:
        # calculate approximate folder size
        dir_size = 0
        for root, _, files in os.walk(d):
            for f in files:
                fp = os.path.join(root, f)
                if not os.path.islink(fp):
                    dir_size += os.path.getsize(fp)
        total_raw_size += dir_size
        print(f"  • {d:<60} [{format_bytes(dir_size)}]")
    print(f"📊 Total uncompressed source size: {format_bytes(total_raw_size)}")

    # 2. Setup output archive path
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{args.name_tag}_{timestamp}.tar.gz"
    archive_path = os.path.abspath(os.path.join(args.output_dir, archive_name))

    # 3. Create tar.gz using system tar (with pigz if available for multi-core speed)
    print(f"\n🗜️  Compressing into: {archive_path}")
    has_pigz = shutil.which("pigz") is not None
    compress_cmd = "pigz" if has_pigz else "gzip"
    if has_pigz:
        print("⚡ Multi-core compressor `pigz` detected! Using parallel compression.")
    else:
        print("ℹ️ Standard gzip compression will be used.")

    tar_cmd = ["tar"]
    if args.exclude_h5:
        print("⚠️ Mode: Excluding .h5 / .keras weight files (backup plots, logs, metrics only).")
        tar_cmd.extend(["--exclude='*.h5'", "--exclude='*.keras'"])
    
    # Use -I for custom compressor, or -czf
    if has_pigz:
        tar_cmd.extend(["-I", "pigz", "-cf", archive_path])
    else:
        tar_cmd.extend(["-czf", archive_path])

    # Append paths
    tar_cmd.extend(resolved_dirs)

    start_time = time.time()
    try:
        # Run tar command
        # Quote arguments safely or pass list directly without shell=True
        quoted_cmd = [f'"{x}"' if ' ' in x else x for x in tar_cmd]
        run_command(' '.join(quoted_cmd))
    except Exception as e:
        print(f"⚠️ System tar encountered an issue: {e}. Falling back to Python tarfile...")
        with tarfile.open(archive_path, "w:gz") as tar:
            for d in resolved_dirs:
                arcname = os.path.basename(d)
                print(f"  Archiving {d} as {arcname}...")
                tar.add(d, arcname=arcname)

    elapsed = time.time() - start_time
    compressed_size = os.path.getsize(archive_path)
    ratio = (1.0 - (compressed_size / total_raw_size)) * 100 if total_raw_size > 0 else 0
    print(f"\n✅ Compression finished in {elapsed:.1f}s!")
    print(f"📦 Archive: {archive_name}")
    print(f"📊 Final Size: {format_bytes(compressed_size)} (Saved {ratio:.1f}% space)")

    # 4. Verify Archive
    print("\n🔍 Verifying archive integrity...")
    try:
        run_command(f"tar -tzf '{archive_path}' | head -n 10")
        print("✅ Archive verified valid and readable!")
    except Exception as e:
        print(f"❌ Verification warning: {e}")

    # 5. Rclone transfer to clara_drive
    if args.skip_upload:
        print("\n⏩ --skip-upload specified. Skipping rclone transfer.")
        print(f"📍 Archive saved locally at: {archive_path}")
        return

    remote = args.rclone_remote
    print(f"\n🚀 Initiating transfer to rclone destination: {remote}")
    check_rclone_remote(remote.split(":")[0])

    rclone_cmd = [
        "rclone", "copy",
        archive_path,
        remote,
        "--progress",
        "--transfers=4",
        "--checkers=8",
        "--retries=5",
        "--low-level-retries=10",
        "--stats=5s"
    ]

    rclone_cmd_str = " ".join(rclone_cmd)
    try:
        run_command(rclone_cmd_str)
        print(f"\n🎉 Transfer complete! Archive successfully uploaded to: {remote}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Rclone transfer failed with exit code {e.returncode}.")
        print("💡 Troubleshooting suggestions:")
        print(f"   1. Test your remote connection: `rclone lsd {remote.split(':')[0]}:`")
        print(f"   2. If using rclone mount, you can copy directly: `cp '{archive_path}' /path/to/mount/`")
        print(f"   3. Manual upload command: `rclone copy '{archive_path}' '{remote}' --progress`")
        sys.exit(e.returncode)

    if not args.keep_local:
        print(f"🧹 Removing local archive {archive_path} (--keep-local=False)")
        os.remove(archive_path)
    else:
        print(f"💾 Local copy preserved at: {archive_path}")

    print("\n" + "=" * 80)
    print("🏆 ALL EXPERIMENTAL RESULTS SECURELY BACKED UP & ARCHIVED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
