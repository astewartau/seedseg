"""Extract one DICOM from each date-named zip and read PatientID + AcquisitionDate."""
import zipfile, sys, os, io

try:
    import pydicom
except ImportError:
    print("Need pydicom")
    sys.exit(1)

drop_folders = [
    "/QRISdata/Q0748/data/2023-Prostate-Drop-Folder",
    "/QRISdata/Q0748/data/2023-Prostate-Drop-Folder2",
]

for folder in drop_folders:
    print(f"\n=== {os.path.basename(folder)} ===")
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.zip'):
            continue
        # Skip subject-named zips (already identified), tars, and partial uploads
        if fname.startswith('z') or '.part' in fname or 'Archive' in fname:
            continue
        # Skip ones we already know are per-subject
        if 'PROSTATE' in fname.upper() and 'Z' in fname.upper():
            continue
            
        fpath = os.path.join(folder, fname)
        try:
            with zipfile.ZipFile(fpath, 'r') as zf:
                # Find first DICOM-like file
                dcm_file = None
                for name in zf.namelist():
                    if name.endswith('/'):
                        continue
                    # Look for DICOM files (often no extension or .dcm)
                    basename = os.path.basename(name)
                    if basename == 'DICOMDIR':
                        continue
                    if '.' not in basename or basename.endswith('.dcm'):
                        dcm_file = name
                        break
                
                if dcm_file is None:
                    print(f"  {fname}: NO DICOM FOUND")
                    continue
                    
                data = zf.read(dcm_file)
                ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
                patient_id = getattr(ds, 'PatientID', 'UNKNOWN')
                patient_name = str(getattr(ds, 'PatientName', 'UNKNOWN'))
                acq_date = getattr(ds, 'AcquisitionDate', getattr(ds, 'StudyDate', 'UNKNOWN'))
                modality = getattr(ds, 'Modality', 'UNKNOWN')
                desc = getattr(ds, 'SeriesDescription', getattr(ds, 'StudyDescription', ''))
                print(f"  {fname}: PatientID={patient_id}, Name={patient_name}, Date={acq_date}, Mod={modality}, Desc={desc}")
        except Exception as e:
            print(f"  {fname}: ERROR - {e}")
