"""
Phase 2: Create organized per-subject DICOM archives on the RDM.

Reads the inventory CSV and creates:
  /QRISdata/Q0748/data/prostate-dicoms-organized/
    {YYYY-MM-DD}_z{ID}/          (one folder per subject+date)
      MR/                        (MR DICOMs)
      CT/                        (CT DICOMs if present)
      RS/                        (RTSTRUCT if present)
      RE/                        (Registration if present)

Then tar.gz each folder into {YYYY-MM-DD}_z{ID}.tar.gz and remove the folder.

Also generates a subject-level summary CSV.
"""
import csv, os, sys, zipfile, shutil, tarfile, re, io
from collections import defaultdict

import pydicom

# ============================================================
# Configuration
# ============================================================
INVENTORY_CSV = "/home/uqaste15/data/2024-prostate/prostate_dicom_inventory.csv"
OUTPUT_DIR = "/QRISdata/Q0748/data/prostate-dicoms-organized"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "subject_summary.csv")
STAGING_DIR = "/home/uqaste15/data/2024-prostate/tmp-staging"

# Skip non-prostate subjects
SKIP_SUBJECTS = {'z3475893', 'z1873164'}  # liver, CEST brain

# ============================================================
# Helpers
# ============================================================
def classify_file(filename):
    """Classify a DICOM filename by type."""
    base = os.path.basename(filename).upper()
    if base.startswith('RS.') or base.startswith('RS_'):
        return 'RS'
    if base.startswith('RE.') or base.startswith('RE_'):
        return 'RE'
    if base.startswith('MR.') or base.startswith('MR_'):
        return 'MR'
    if base.startswith('CT.') or base.startswith('CT_'):
        return 'CT'
    # .IMA files - check content of name
    if '.CT.' in base or base.startswith('CT'):
        return 'CT'
    if '.MR.' in base:
        return 'MR'
    return None


def extract_subject_from_filename(filename):
    """Extract z-ID from filename."""
    match = re.search(r'[Zz](\d{6,7})', filename)
    if match:
        digits = match.group(1).zfill(7)
        return f"z{digits}"
    return None


def read_dicom_date(data):
    """Read acquisition date from DICOM binary data."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        date = getattr(ds, 'AcquisitionDate', '') or getattr(ds, 'StudyDate', '')
        if date and len(date) == 8:
            return f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    except:
        pass
    return None


# ============================================================
# Load inventory
# ============================================================
print("Loading inventory...")
records = []
with open(INVENTORY_CSV) as f:
    for row in csv.DictReader(f):
        if row['subject_id'] in SKIP_SUBJECTS:
            continue
        records.append(row)

# Group by source file to process each zip once
by_source = defaultdict(list)
for r in records:
    key = (r['source_folder'], r['source_file'])
    by_source[key].append(r)

# Track what we've already organized to avoid duplicates
# Key: (subject_id, acq_date, modality_type)
organized = set()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)

# Map source_folder names back to full paths
FOLDER_PATHS = {
    "2023-Prostate-Drop-Folder": "/QRISdata/Q0748/data/2023-Prostate-Drop-Folder",
    "2023-Prostate-Drop-Folder2": "/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2",
    "2025-Prostate-Drop-Folder": "/QRISdata/Q0748/data/2025-Prostate-Drop-Folder",
    "2026-Prostate-Drop-Folder": "/QRISdata/Q0748/data/2026-Prostate-Drop-Folder",
    "data/test/dicoms": "/home/uqaste15/data/2024-prostate/data/test/dicoms",
}

# ============================================================
# Process each source
# ============================================================
# subject_data tracks best source for each subject
# Prefer sources with more modalities (MR+CT+RS+RE > MR+RS+RE > MR only)
subject_best = defaultdict(lambda: {'n_total': 0, 'records': []})

for (src_folder, src_file), recs in by_source.items():
    for r in recs:
        subj = r['subject_id']
        n = int(r['n_total'])
        # Accumulate all records for each subject
        subject_best[subj]['records'].append(r)

# For each subject, find the best source (most files, prefer has_rs + has_re)
print(f"\nProcessing {len(subject_best)} subjects...")

subject_summary = []
processed = 0
errors = 0

for subj_id in sorted(subject_best.keys()):
    recs = subject_best[subj_id]['records']

    # Sort records: prefer sources with RS+RE, then most files
    def score(r):
        s = 0
        if r['has_rtstruct'] == 'True': s += 1000
        if r['has_registration'] == 'True': s += 500
        s += int(r['n_mr']) * 10
        s += int(r['n_ct']) * 5
        return s

    # Get best record per source (highest score)
    recs.sort(key=score, reverse=True)
    best = recs[0]

    src_folder = best['source_folder']
    src_file = best['source_file']
    acq_date = best['acq_date']

    if acq_date == 'unknown':
        # Try other records for this subject
        for r in recs:
            if r['acq_date'] != 'unknown':
                acq_date = r['acq_date']
                break

    # Create archive name
    archive_name = f"{acq_date}_{subj_id}" if acq_date != 'unknown' else f"unknown-date_{subj_id}"
    staging_path = os.path.join(STAGING_DIR, archive_name)
    tar_path = os.path.join(OUTPUT_DIR, f"{archive_name}.tar.gz")

    if os.path.exists(tar_path):
        print(f"  [{subj_id}] Already exists: {archive_name}.tar.gz")
        subject_summary.append({
            'subject_id': subj_id,
            'acq_date': acq_date,
            'archive_name': f"{archive_name}.tar.gz",
            'has_mr': int(best['n_mr']) > 0,
            'has_ct': int(best['n_ct']) > 0,
            'has_rtstruct': best['has_rtstruct'] == 'True',
            'has_registration': best['has_registration'] == 'True',
            'source': f"{src_folder}/{src_file}",
            'in_train': best['in_train'] == 'True',
            'in_test': best['in_test'] == 'True',
            'status': 'exists',
        })
        processed += 1
        continue

    print(f"  [{subj_id}] Creating {archive_name}.tar.gz from {src_folder}/{src_file}")

    try:
        # Clean staging
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path)
        os.makedirs(staging_path, exist_ok=True)

        full_src_path = FOLDER_PATHS.get(src_folder, '')

        if src_folder == "data/test/dicoms":
            # Local directory source
            src_dir = os.path.join(full_src_path, src_file)
            for fname in os.listdir(src_dir):
                fpath = os.path.join(src_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                ftype = classify_file(fname) or 'OTHER'
                dest_dir = os.path.join(staging_path, ftype)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(fpath, os.path.join(dest_dir, fname))

                # Try to get date if still unknown
                if acq_date == 'unknown' and fname.endswith('.dcm') and ftype == 'MR':
                    try:
                        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                        d = getattr(ds, 'AcquisitionDate', '') or getattr(ds, 'StudyDate', '')
                        if d and len(d) == 8:
                            acq_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                    except:
                        pass

        else:
            # Zip source
            zip_path = os.path.join(full_src_path, src_file)
            if not os.path.exists(zip_path):
                print(f"    ERROR: Zip not found: {zip_path}")
                errors += 1
                continue

            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('/'):
                        continue

                    # Check if this file belongs to our subject
                    file_subj = extract_subject_from_filename(name)
                    if file_subj != subj_id:
                        # For scan_day zips, include all files
                        if best.get('struct_type') != 'scan_day' and file_subj is not None:
                            continue

                    base = os.path.basename(name)
                    ftype = classify_file(name) or 'OTHER'
                    dest_dir = os.path.join(staging_path, ftype)
                    os.makedirs(dest_dir, exist_ok=True)

                    # Extract file
                    data = zf.read(name)
                    dest_file = os.path.join(dest_dir, base)
                    # Avoid overwriting
                    if os.path.exists(dest_file):
                        root, ext = os.path.splitext(base)
                        dest_file = os.path.join(dest_dir, f"{root}_dup{ext}")
                    with open(dest_file, 'wb') as f:
                        f.write(data)

                    # Try to get date if still unknown
                    if acq_date == 'unknown' and ftype == 'MR':
                        d = read_dicom_date(data)
                        if d:
                            acq_date = d

        # If we found the date during extraction, rename
        if acq_date != 'unknown' and archive_name.startswith('unknown-date_'):
            new_archive_name = f"{acq_date}_{subj_id}"
            new_staging = os.path.join(STAGING_DIR, new_archive_name)
            new_tar_path = os.path.join(OUTPUT_DIR, f"{new_archive_name}.tar.gz")
            if not os.path.exists(new_staging):
                os.rename(staging_path, new_staging)
                staging_path = new_staging
                archive_name = new_archive_name
                tar_path = new_tar_path

        # Count files in staging
        file_count = sum(len(files) for _, _, files in os.walk(staging_path))
        if file_count == 0:
            print(f"    WARNING: No files extracted for {subj_id}")
            shutil.rmtree(staging_path, ignore_errors=True)
            errors += 1
            continue

        # Create tar.gz
        with tarfile.open(tar_path, 'w:gz') as tf:
            tf.add(staging_path, arcname=archive_name)

        tar_size_mb = os.path.getsize(tar_path) / 1024 / 1024
        print(f"    Created: {archive_name}.tar.gz ({tar_size_mb:.1f} MB, {file_count} files)")

        # Clean up staging
        shutil.rmtree(staging_path, ignore_errors=True)

        # Check what modalities we got
        has_mr = os.path.exists(os.path.join(staging_path, 'MR')) if os.path.exists(staging_path) else int(best['n_mr']) > 0
        has_ct = os.path.exists(os.path.join(staging_path, 'CT')) if os.path.exists(staging_path) else int(best['n_ct']) > 0

        subject_summary.append({
            'subject_id': subj_id,
            'acq_date': acq_date,
            'archive_name': f"{archive_name}.tar.gz",
            'has_mr': int(best['n_mr']) > 0,
            'has_ct': int(best['n_ct']) > 0,
            'has_rtstruct': best['has_rtstruct'] == 'True',
            'has_registration': best['has_registration'] == 'True',
            'source': f"{src_folder}/{src_file}",
            'in_train': best['in_train'] == 'True',
            'in_test': best['in_test'] == 'True',
            'status': 'created',
        })
        processed += 1

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()
        errors += 1
        # Clean up on error
        shutil.rmtree(staging_path, ignore_errors=True)

# Also process additional records (other scan dates / modalities)
# TODO: handle multiple acquisitions per subject

# ============================================================
# Write summary CSV
# ============================================================
summary_fields = ['subject_id', 'acq_date', 'archive_name', 'has_mr', 'has_ct',
                   'has_rtstruct', 'has_registration', 'source', 'in_train', 'in_test', 'status']
subject_summary.sort(key=lambda x: x['subject_id'])

with open(SUMMARY_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(subject_summary)

print(f"\n{'='*60}")
print(f"REORGANIZATION COMPLETE")
print(f"{'='*60}")
print(f"Processed: {processed}")
print(f"Errors: {errors}")
print(f"Summary CSV: {SUMMARY_CSV}")
print(f"Archives in: {OUTPUT_DIR}")

# Clean up staging
shutil.rmtree(STAGING_DIR, ignore_errors=True)
