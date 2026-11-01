"""QC montage: synthetic cases (axial slices through the markers) next to one real image."""
import json, sys
import nibabel as nib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out, cases, real, png = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]
cols = []
for c in cases:
    v = np.asarray(nib.load(f"{out}/images/{c}.nii.gz").dataobj, np.float32)
    l = np.asarray(nib.load(f"{out}/labels/{c}.nii.gz").dataobj)
    ref = np.median(v[l == 2]); ks = sorted({int(round(m["center_vox"][2])) for m in json.load(open(f"{out}/params/{c}.json"))["markers"]})
    cols.append((c, v / ref, l, [ks[0], v.shape[2] // 2 - 8, v.shape[2] // 2 + 6]))
if real:
    v = np.asarray(nib.load(real).dataobj, np.float32); l = np.asarray(nib.load(real.replace("images", "labels/corrected") + ".gz").dataobj)
    ref = np.median(v[l == 2]); cz = int(np.median(np.nonzero(l == 1)[2]))
    cols.append(("real", v / ref, l, [cz, v.shape[2] // 2 - 8, v.shape[2] // 2 + 6]))
fig, ax = plt.subplots(3, len(cols), figsize=(6 * len(cols), 18))
for j, (name, v, l, ks) in enumerate(cols):
    for i, k in enumerate(ks):
        a = ax[i, j]; a.imshow(v[:, :, k].T, cmap="gray", vmin=0, vmax=1.8); a.set_title(f"{name} z={k}"); a.axis("off")
        if i == 0: a.contour((l[:, :, k] == 1).T, [0.5], colors="r", linewidths=0.6)
plt.tight_layout(); plt.savefig(png, dpi=50)
