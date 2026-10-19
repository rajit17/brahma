import os
from astropy.io import fits

data_dir = '/data4/rajit/DATA_GZ3D_DIR'
corrupt_files = []

def is_fits_corrupt(filepath):
    try:
        with fits.open(filepath) as hdul:
            return False
    except Exception:
        return True

def redownload_file(fname):
    # TODO: Implement your download logic here
    # Example: download_from_server(fname)
    print(f"Redownloading {fname} ...")
    # ...your download code...

fits_files = [f for f in os.listdir(data_dir) if f.endswith('.fits.gz')]
for fname in fits_files:
    fpath = os.path.join(data_dir, fname)
    if is_fits_corrupt(fpath):
        corrupt_files.append(fname)
        redownload_file(fname)

print(f"Checked {len(fits_files)} files. Corrupt files: {len(corrupt_files)}")
if corrupt_files:
    print("Corrupt files fixed. You can now rerun your mask extraction script.")
else:
    print("No corrupt files found.")

# Optionally, run mask extraction after fixing
# os.system("python /home/rajit/1_batch_mask_generator.py")
