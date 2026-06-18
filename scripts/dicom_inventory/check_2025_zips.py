"""Check 2025 drop folder zips for PatientID + AcquisitionDate."""
import zipfile, os, io
import pydicom

folder = "/QRISdata/Q0748/data/2025-Prostate-Drop-Folder"
for fname in sorted(os.listdir(folder)):
    if not fname.endswith('.zip') or fname.startswith('prostate-backup'):
        continue
    fpath = os.path.join(folder, fname)
    try:
        with zipfile.ZipFile(fpath, 'r') as zf:
            dcm_file = None
            for name in zf.namelist():
                if name.endswith('/') or os.path.basename(name) == 'DICOMDIR':
                    continue
                basename = os.path.basename(name)
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
            print(f"{fname}: PatientID={patient_id}, Name={patient_name}, Date={acq_date}, Mod={modality}, Desc={desc}")
    except Exception as e:
        print(f"{fname}: ERROR - {e}")
