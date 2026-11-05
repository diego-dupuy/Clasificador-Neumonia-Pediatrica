"""Read DICOM files (the standard format of hospital imaging / PACS).

The patient's identifying data is removed before anything else is done with the
file, and the pixel data is converted to an 8-bit image the model can use.
"""

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_voi_lut

# Identifying attributes removed (subset of the DICOM PS3.15 Basic Profile)
IDENTIFYING_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
    "PatientAddress", "OtherPatientIDs", "OtherPatientNames", "PatientTelephoneNumbers",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
    "PerformingPhysicianName", "OperatorsName", "AccessionNumber", "StudyID",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "DeviceSerialNumber",
]


def anonymize(ds):
    """Remove identifying attributes and private tags in place. Returns how many were removed."""
    removed = 0
    for tag in IDENTIFYING_TAGS:
        if tag in ds:
            delattr(ds, tag)
            removed += 1
    before = len(ds)
    ds.remove_private_tags()  # manufacturer-specific tags can also contain identifying data
    return removed + (before - len(ds))


def read_dicom(path):
    """Return (anonymized dataset, PIL image, number of attributes removed)."""
    ds = pydicom.dcmread(path)
    removed = anonymize(ds)

    pixels = ds.pixel_array.astype(np.float32)
    pixels = apply_voi_lut(pixels, ds)  # apply the display window stored in the file
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels  # MONOCHROME1 stores bright as low values: invert
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-8) * 255
    image = Image.fromarray(pixels.astype(np.uint8)).convert("RGB")
    return ds, image, removed
