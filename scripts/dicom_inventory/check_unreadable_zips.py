"""Investigate zips that returned NO DICOM FOUND - list file extensions and structure."""
import zipfile, os, io
from collections import Counter

try:
    import pydicom
except ImportError:
    pass

folders_and_zips = {
    "/QRISdata/Q0748/data/2023-Prostate-Drop-Folder": [
        "QSM2_130323.zip", "QSM_071122.zip", "QSM_080323.zip",
        "QSM_130323.zip", "QSM_221222.zip",
        "QSM_CT_240823.zip", "QSM_CT_241023.zip", "QSM_CT_250823.zip",
    ],
    "/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2": [
        "CT.zip", "CT_QSM_050723.zip", "CT_QSM_140623.zip",
        "CT_QSM_190623.zip", "CT_QSM_230523.zip",
        "QSM_050723.zip", "QSM_080523.zip",
        "QSM_140623.zip", "QSM_190523.zip", "QSM_190623.zip",
        "QSM_240523.zip", "QSM_270723.zip",
        "QSM_CT_190523.zip", "QSM_CT_240523.zip", "QSM_CT_240823.zip",
        "QSM_CT_241023.zip", "QSM_CT_250823.zip",
        "QSM_CT_270723.zip", "QSM_CT_310723.zip",
        "QSM2_130323.zip", "QSM_071122.zip", "QSM_080323.zip",
        "QSM_130323.zip", "QSM_221222.zip",
    ],
}

for folder, zips in folders_and_zips.items():
    folder_name = os.path.basename(folder)
    print(f"\n{'='*60}")
    print(f"=== {folder_name} ===")
    print(f"{'='*60}")

    for zname in sorted(set(zips)):
        fpath = os.path.join(folder, zname)
        if not os.path.exists(fpath):
            print(f"\n  {zname}: FILE NOT FOUND")
            continue

        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"\n  {zname} ({size_mb:.1f} MB):")

        try:
            with zipfile.ZipFile(fpath, 'r') as zf:
                names = zf.namelist()
                # Count extensions
                exts = Counter()
                dirs = set()
                for name in names:
                    if name.endswith('/'):
                        dirs.add(name.rstrip('/'))
                        continue
                    _, ext = os.path.splitext(name)
                    exts[ext if ext else '(no ext)'] = exts.get(ext if ext else '(no ext)', 0) + 1

                print(f"    Files: {len([n for n in names if not n.endswith('/')])} | Dirs: {len(dirs)}")
                print(f"    Extensions: {dict(exts)}")

                # Show first few top-level entries
                top_level = sorted(set(n.split('/')[0] for n in names if n))
                if len(top_level) <= 5:
                    print(f"    Top-level: {top_level}")
                else:
                    print(f"    Top-level ({len(top_level)}): {top_level[:5]}...")

                # Show a few sample file paths
                sample_files = [n for n in names if not n.endswith('/')][:5]
                print(f"    Sample files: {sample_files}")

                # Try reading first non-directory file as DICOM
                for name in names:
                    if name.endswith('/'):
                        continue
                    basename = os.path.basename(name)
                    # Skip known non-DICOM
                    if basename.endswith(('.zip', '.tar', '.gz', '.txt', '.xml', '.pdf', '.png', '.jpg')):
                        continue
                    try:
                        data = zf.read(name)
                        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
                        patient_id = getattr(ds, 'PatientID', 'UNKNOWN')
                        acq_date = getattr(ds, 'AcquisitionDate', getattr(ds, 'StudyDate', 'UNKNOWN'))
                        modality = getattr(ds, 'Modality', 'UNKNOWN')
                        print(f"    >>> DICOM FOUND: PatientID={patient_id}, Date={acq_date}, Mod={modality}")
                        print(f"        File: {name}")
                        break
                    except:
                        continue

        except zipfile.BadZipFile:
            print(f"    BAD ZIP FILE (corrupt or not a zip)")
        except Exception as e:
            print(f"    ERROR: {e}")
