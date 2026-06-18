"""Catalog backup tar/zip contents to understand what's duplicated."""
import tarfile, zipfile, os
from collections import Counter

backups = [
    ("/QRISdata/Q0748/data/2025-Prostate-Drop-Folder/prostate-backup.tar", "tar"),
    ("/QRISdata/Q0748/data/2025-Prostate-Drop-Folder/prostate-backup-2.tar", "tar"),
    ("/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2/2024-prostate.tar", "tar"),
    ("/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2/models-paper.tar", "tar"),
    ("/QRISdata/Q0748/data/2023-2016-Jonathan-Prostate/2023-prostate.tar", "tar"),
    ("/QRISdata/Q0748/data/2023-2016-Jonathan-Prostate/2023-2016-Jonathan-Prostate-Bids.tar", "tar"),
    ("/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2/2024-05-21-prostate-bids.tar", "tar"),
    ("/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2/2023-01-18-new-data.tar", "tar"),
]

for fpath, ftype in backups:
    fname = os.path.basename(fpath)
    folder = os.path.basename(os.path.dirname(fpath))
    size_gb = os.path.getsize(fpath) / 1024**3 if os.path.exists(fpath) else 0

    print(f"\n{'='*70}")
    print(f"{folder}/{fname} ({size_gb:.1f} GB)")
    print(f"{'='*70}")

    if not os.path.exists(fpath):
        print("  FILE NOT FOUND")
        continue

    try:
        if ftype == "tar":
            with tarfile.open(fpath, 'r') as tf:
                members = tf.getmembers()
                files = [m for m in members if m.isfile()]
                dirs = [m for m in members if m.isdir()]

                # Count extensions
                exts = Counter()
                for m in files:
                    _, ext = os.path.splitext(m.name)
                    exts[ext if ext else '(no ext)'] = exts.get(ext if ext else '(no ext)', 0) + 1

                # Top-level structure
                top_level = sorted(set(m.name.split('/')[0] for m in members if m.name))
                second_level = sorted(set('/'.join(m.name.split('/')[:2]) for m in members if '/' in m.name))[:20]

                print(f"  Files: {len(files)} | Dirs: {len(dirs)}")
                total_size = sum(m.size for m in files)
                print(f"  Total uncompressed: {total_size / 1024**3:.1f} GB")
                print(f"  Extensions: {dict(exts.most_common(10))}")
                print(f"  Top-level: {top_level}")
                if second_level:
                    print(f"  Second-level (first 20): {second_level}")

                # Look for subject IDs
                import re
                subject_ids = set()
                for m in members:
                    matches = re.findall(r'z\d{7}', m.name)
                    subject_ids.update(matches)
                if subject_ids:
                    print(f"  Subject IDs found ({len(subject_ids)}): {sorted(subject_ids)[:20]}{'...' if len(subject_ids) > 20 else ''}")

                # Check for model files
                model_files = [m.name for m in files if m.name.endswith('.pth')]
                if model_files:
                    print(f"  Model checkpoints ({len(model_files)}): {model_files[:5]}{'...' if len(model_files) > 5 else ''}")

                # Check for scripts
                script_files = [m.name for m in files if m.name.endswith('.py') or m.name.endswith('.sh')]
                if script_files:
                    print(f"  Scripts ({len(script_files)}): {script_files[:10]}{'...' if len(script_files) > 10 else ''}")

    except Exception as e:
        print(f"  ERROR: {e}")
