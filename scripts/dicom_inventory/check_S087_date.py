import pydicom
ds = pydicom.dcmread("data/test/dicoms/S087/RS.S087.sCT_upscale.dcm", stop_before_pixels=True)
for attr in ["StudyDate", "AcquisitionDate", "ContentDate", "InstanceCreationDate", "StructureSetDate"]:
    print(f"{attr}: {getattr(ds, attr, 'N/A')}")
