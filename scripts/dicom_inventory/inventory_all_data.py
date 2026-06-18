"""
Phase 1: Build a comprehensive inventory of all prostate DICOM data across the RDM.

Scans all data sources, reads DICOM headers, and produces a CSV documenting:
- Subject ID, acquisition date, modalities, series descriptions
- Whether RS (RTSTRUCT), RE (registration) files are present
- Source location
- Whether the subject is in train/test splits

Output: prostate_dicom_inventory.csv
"""
import zipfile, os, io, sys, csv, re, traceback
from collections import defaultdict

import pydicom

# ============================================================
# Configuration
# ============================================================
OUTPUT_CSV = "/home/uqaste15/data/2024-prostate/prostate_dicom_inventory.csv"

# All zip data sources: (source_folder, zip_file, structure_type)
# structure_type:
#   "per_subject" = zip contains z{ID}/ folders with mixed modality DICOMs
#   "scan_day"    = zip contains a single scan-day export (one subject, one or more series)
#   "nested_per_subject" = zip contains a top-level folder then z{ID}/ subfolders

SOURCES = []

# --- Per-subject archives ---
rdm = "/QRISdata/Q0748/data"

SOURCES.append((f"{rdm}/2025-Prostate-Drop-Folder", "GS data sets.zip", "nested_per_subject"))
SOURCES.append((f"{rdm}/2025-Prostate-Drop-Folder", "Gold seed data sets for JG.zip", "nested_per_subject"))
SOURCES.append((f"{rdm}/2025-Prostate-Drop-Folder", "NEW_retro_DATA_04.06.25.zip", "nested_per_subject"))
SOURCES.append((f"{rdm}/2026-Prostate-Drop-Folder", "fresh data for Ashley (JG).zip", "nested_per_subject"))

# --- Date-named scan-day zips (Drop Folder 1) ---
df1 = f"{rdm}/2023-Prostate-Drop-Folder"
for fname in sorted(os.listdir(df1)):
    if not fname.endswith('.zip'):
        continue
    if fname.startswith('Archive'):
        continue
    # Skip per-subject zips that have Z-ID and PROSTATE in the name with a subject
    SOURCES.append((df1, fname, "scan_day"))

# --- Date-named scan-day zips (Drop Folder 2) ---
df2 = f"{rdm}/2023-Prostate-Drop-Folder2"
for fname in sorted(os.listdir(df2)):
    if not fname.endswith('.zip'):
        continue
    if fname.startswith('Archive') or fname.endswith('.part'):
        continue
    # Skip tars and known non-zip
    SOURCES.append((df2, fname, "scan_day"))

# --- 2025 individual zips ---
df25 = f"{rdm}/2025-Prostate-Drop-Folder"
for fname in sorted(os.listdir(df25)):
    if not fname.endswith('.zip'):
        continue
    # Skip the per-subject archives and backups we already handle
    if fname in ("GS data sets.zip", "Gold seed data sets for JG.zip",
                 "NEW_retro_DATA_04.06.25.zip", "z0772718_structure.zip"):
        continue
    if 'backup' in fname.lower():
        continue
    SOURCES.append((df25, fname, "scan_day"))

# --- 2026 folder ---
df26 = f"{rdm}/2026-Prostate-Drop-Folder"
for fname in sorted(os.listdir(df26)):
    if not fname.endswith('.zip'):
        continue
    if fname == "fresh data for Ashley (JG).zip":
        continue  # Already added above
    SOURCES.append((df26, fname, "scan_day"))

# --- Also scan existing test DICOMs on disk ---
LOCAL_TEST_DICOMS = "/home/uqaste15/data/2024-prostate/data/test/dicoms"

# ============================================================
# Load train/test split info
# ============================================================
train_dir = "/home/uqaste15/data/2024-prostate/data/train"
test_dicom_dir = "/home/uqaste15/data/2024-prostate/data/test/dicoms"

train_subjects = set()
for d in os.listdir(train_dir):
    if d.startswith('z'):
        train_subjects.add(d.split('-')[0].lower())

test_subjects = set()
for d in os.listdir(test_dicom_dir):
    if d.startswith('z'):
        test_subjects.add(d.lower())


def normalize_subject_id(raw_id):
    """Normalize subject ID to lowercase z{7digits} format."""
    raw_id = raw_id.strip()
    # Handle uppercase Z
    if raw_id.upper().startswith('Z'):
        digits = raw_id[1:]
        # Pad to 7 digits
        digits = digits.zfill(7)
        return f"z{digits}"
    return raw_id.lower()


def classify_file(filename):
    """Classify a DICOM filename by type: MR, CT, RS, RE, or OTHER."""
    base = os.path.basename(filename).upper()
    if base.startswith('RS.') or base.startswith('RS_'):
        return 'RS'
    if base.startswith('RE.') or base.startswith('RE_'):
        return 'RE'
    if base.startswith('MR.') or base.startswith('MR_'):
        return 'MR'
    if base.startswith('CT.') or base.startswith('CT_'):
        return 'CT'
    return None  # Will need DICOM header to classify


def read_dicom_info(data):
    """Read key fields from DICOM binary data."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        return {
            'patient_id': str(getattr(ds, 'PatientID', '')),
            'patient_name': str(getattr(ds, 'PatientName', '')),
            'acq_date': getattr(ds, 'AcquisitionDate', '') or getattr(ds, 'StudyDate', '') or '',
            'modality': getattr(ds, 'Modality', ''),
            'series_desc': getattr(ds, 'SeriesDescription', '') or getattr(ds, 'StudyDescription', '') or '',
            'study_date': getattr(ds, 'StudyDate', ''),
        }
    except Exception:
        return None


def format_date(raw_date):
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if raw_date and len(raw_date) == 8:
        return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    return raw_date or 'unknown'


def extract_subject_from_filename(filename):
    """Try to extract z-ID from a filename."""
    match = re.search(r'[Zz](\d{6,7})', filename)
    if match:
        return normalize_subject_id(f"z{match.group(1)}")
    return None


def extract_date_from_rs_filename(filename):
    """Extract date from RS filename like RS.z{ID}.sCT{DDMMYY}.dcm or CT_{DDMMYY}."""
    base = os.path.basename(filename)
    # Match patterns: sCT300425, CT300425, CT_300425, sCT_140825
    match = re.search(r's?CT_?(\d{6})', base, re.IGNORECASE)
    if match:
        ddmmyy = match.group(1)
        dd, mm, yy = ddmmyy[:2], ddmmyy[2:4], ddmmyy[4:6]
        year = int(yy)
        if year > 50:
            year += 1900
        else:
            year += 2000
        try:
            return f"{year:04d}-{mm}-{dd}"
        except:
            pass
    return None


# ============================================================
# Main inventory
# ============================================================
# Each row: subject_id, acq_date, modality, series_desc, has_rs, has_re,
#           n_mr, n_ct, n_rs, n_re, n_other, source_folder, source_file,
#           in_train, in_test

# Group by (source_file, subject_id, acq_date) to aggregate
inventory = []  # list of dicts

# Track seen to deduplicate
seen_sources = set()

print("Starting inventory scan...")
print(f"Total sources to scan: {len(SOURCES)}")

for idx, (src_folder, src_file, struct_type) in enumerate(SOURCES):
    src_key = os.path.join(src_folder, src_file)

    # Skip exact duplicates across folders
    if src_key in seen_sources:
        continue
    seen_sources.add(src_key)

    fpath = os.path.join(src_folder, src_file)
    if not os.path.exists(fpath):
        print(f"  [{idx+1}/{len(SOURCES)}] SKIP (not found): {src_file}")
        continue

    print(f"  [{idx+1}/{len(SOURCES)}] Scanning: {os.path.basename(src_folder)}/{src_file}")

    try:
        with zipfile.ZipFile(fpath, 'r') as zf:
            names = zf.namelist()
            files = [n for n in names if not n.endswith('/')]

            if struct_type in ("per_subject", "nested_per_subject"):
                # Group files by subject (z-ID extracted from path)
                subject_files = defaultdict(list)
                for name in files:
                    subj = extract_subject_from_filename(name)
                    if subj:
                        subject_files[subj].append(name)

                for subj_id, sfiles in sorted(subject_files.items()):
                    # Count file types by filename prefix
                    counts = {'MR': 0, 'CT': 0, 'RS': 0, 'RE': 0, 'OTHER': 0}
                    series_descs = set()
                    acq_date = None
                    modalities = set()

                    for sf in sfiles:
                        ftype = classify_file(sf)
                        if ftype:
                            counts[ftype] += 1
                            modalities.add(ftype)
                        else:
                            counts['OTHER'] += 1

                    # Try to get date from RS filename first (most reliable for anonymized data)
                    rs_files = [sf for sf in sfiles if classify_file(sf) == 'RS']
                    for sf in rs_files:
                        d = extract_date_from_rs_filename(sf)
                        if d:
                            acq_date = d
                            break

                    # Read DICOMs to get date and series info
                    # Prioritize MR files, then CT, then anything else
                    mr_files = [sf for sf in sfiles if classify_file(sf) == 'MR']
                    ct_files = [sf for sf in sfiles if classify_file(sf) == 'CT']
                    other_dcm = [sf for sf in sfiles if classify_file(sf) is None
                                 and (os.path.basename(sf).endswith(('.dcm', '.IMA'))
                                      or '.' not in os.path.basename(sf))]

                    for sf in (mr_files[:1] + ct_files[:1] + other_dcm[:1]):
                        try:
                            data = zf.read(sf)
                            info = read_dicom_info(data)
                            if info:
                                if info['acq_date'] and not acq_date:
                                    acq_date = format_date(info['acq_date'])
                                elif info['study_date'] and not acq_date:
                                    acq_date = format_date(info['study_date'])
                                if info['series_desc']:
                                    series_descs.add(info['series_desc'])
                                if info['modality']:
                                    modalities.add(info['modality'])
                        except:
                            continue

                    inventory.append({
                        'subject_id': subj_id,
                        'acq_date': acq_date or 'unknown',
                        'modalities': ';'.join(sorted(modalities)),
                        'series_descriptions': ';'.join(sorted(series_descs)),
                        'has_rtstruct': counts['RS'] > 0,
                        'has_registration': counts['RE'] > 0,
                        'n_mr': counts['MR'],
                        'n_ct': counts['CT'],
                        'n_rs': counts['RS'],
                        'n_re': counts['RE'],
                        'n_other': counts['OTHER'],
                        'n_total': sum(counts.values()),
                        'source_folder': os.path.basename(src_folder),
                        'source_file': src_file,
                        'in_train': subj_id in train_subjects,
                        'in_test': subj_id in test_subjects,
                    })

            elif struct_type == "scan_day":
                # Single scan-day zip: try to identify subject from filenames or DICOM header
                subj_id = None
                acq_date = None
                modalities = set()
                series_descs = set()
                counts = {'MR': 0, 'CT': 0, 'RS': 0, 'RE': 0, 'OTHER': 0}

                # Try filename-based subject extraction
                for sf in files[:5]:
                    subj_id = extract_subject_from_filename(sf)
                    if subj_id:
                        break

                # Count file types and read one DICOM
                dicom_read = False
                for sf in files:
                    base = os.path.basename(sf)
                    ftype = classify_file(sf)
                    if ftype:
                        counts[ftype] += 1
                        modalities.add(ftype)
                    else:
                        counts['OTHER'] += 1

                    if not dicom_read and (base.endswith(('.dcm', '.IMA')) or
                                           ('.' not in base and not base.startswith('.'))):
                        try:
                            data = zf.read(sf)
                            info = read_dicom_info(data)
                            if info:
                                if not subj_id and info['patient_id']:
                                    subj_id = normalize_subject_id(info['patient_id'])
                                if info['acq_date']:
                                    acq_date = format_date(info['acq_date'])
                                if info['modality']:
                                    modalities.add(info['modality'])
                                if info['series_desc']:
                                    series_descs.add(info['series_desc'])
                                dicom_read = True
                        except:
                            pass

                if subj_id and subj_id.startswith('z'):
                    inventory.append({
                        'subject_id': subj_id,
                        'acq_date': acq_date or 'unknown',
                        'modalities': ';'.join(sorted(modalities)),
                        'series_descriptions': ';'.join(sorted(series_descs)),
                        'has_rtstruct': counts['RS'] > 0,
                        'has_registration': counts['RE'] > 0,
                        'n_mr': counts['MR'],
                        'n_ct': counts['CT'],
                        'n_rs': counts['RS'],
                        'n_re': counts['RE'],
                        'n_other': counts['OTHER'],
                        'n_total': sum(counts.values()),
                        'source_folder': os.path.basename(src_folder),
                        'source_file': src_file,
                        'in_train': subj_id in train_subjects,
                        'in_test': subj_id in test_subjects,
                    })
                else:
                    print(f"    WARNING: Could not identify subject in {src_file} (got: {subj_id})")

    except zipfile.BadZipFile:
        print(f"    ERROR: Bad zip file: {src_file}")
    except Exception as e:
        print(f"    ERROR processing {src_file}: {e}")
        traceback.print_exc()

# --- Also scan local test DICOMs ---
print(f"\n  Scanning local test DICOMs: {LOCAL_TEST_DICOMS}")
for subj_dir in sorted(os.listdir(LOCAL_TEST_DICOMS)):
    subj_path = os.path.join(LOCAL_TEST_DICOMS, subj_dir)
    if not os.path.isdir(subj_path) or not subj_dir.startswith('z'):
        continue

    counts = {'MR': 0, 'CT': 0, 'RS': 0, 'RE': 0, 'OTHER': 0}
    acq_date = None
    modalities = set()
    series_descs = set()

    for fname in os.listdir(subj_path):
        fpath_file = os.path.join(subj_path, fname)
        if not os.path.isfile(fpath_file):
            continue

        ftype = classify_file(fname)
        if ftype:
            counts[ftype] += 1
            modalities.add(ftype)
        else:
            counts['OTHER'] += 1

        # Extract date from RS filename
        if ftype == 'RS' and not acq_date:
            d = extract_date_from_rs_filename(fname)
            if d:
                acq_date = d

    inventory.append({
        'subject_id': subj_dir.lower(),
        'acq_date': acq_date or 'unknown',
        'modalities': ';'.join(sorted(modalities)),
        'series_descriptions': ';'.join(sorted(series_descs)),
        'has_rtstruct': counts['RS'] > 0,
        'has_registration': counts['RE'] > 0,
        'n_mr': counts['MR'],
        'n_ct': counts['CT'],
        'n_rs': counts['RS'],
        'n_re': counts['RE'],
        'n_other': counts['OTHER'],
        'n_total': sum(counts.values()),
        'source_folder': 'data/test/dicoms',
        'source_file': subj_dir,
        'in_train': subj_dir in train_subjects,
        'in_test': True,
    })

# ============================================================
# Write CSV
# ============================================================
fieldnames = ['subject_id', 'acq_date', 'modalities', 'series_descriptions',
              'has_rtstruct', 'has_registration',
              'n_mr', 'n_ct', 'n_rs', 'n_re', 'n_other', 'n_total',
              'source_folder', 'source_file', 'in_train', 'in_test']

inventory.sort(key=lambda x: (x['subject_id'], x['acq_date']))

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(inventory)

# ============================================================
# Summary stats
# ============================================================
unique_subjects = set(r['subject_id'] for r in inventory)
subjects_with_rs = set(r['subject_id'] for r in inventory if r['has_rtstruct'])
subjects_with_re = set(r['subject_id'] for r in inventory if r['has_registration'])
subjects_with_mr = set(r['subject_id'] for r in inventory if r['n_mr'] > 0)
subjects_with_ct = set(r['subject_id'] for r in inventory if r['n_ct'] > 0)

print(f"\n{'='*60}")
print(f"INVENTORY COMPLETE")
print(f"{'='*60}")
print(f"Total records: {len(inventory)}")
print(f"Unique subjects: {len(unique_subjects)}")
print(f"  With MR data: {len(subjects_with_mr)}")
print(f"  With CT data: {len(subjects_with_ct)}")
print(f"  With RTSTRUCT: {len(subjects_with_rs)}")
print(f"  With Registration: {len(subjects_with_re)}")
print(f"  In train set: {len(unique_subjects & train_subjects)}")
print(f"  In test set: {len(unique_subjects & test_subjects)}")
print(f"  Neither: {len(unique_subjects - train_subjects - test_subjects)}")
print(f"\nCSV written to: {OUTPUT_CSV}")
