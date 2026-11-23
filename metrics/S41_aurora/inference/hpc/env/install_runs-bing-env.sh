module avail frameworks >/dev/null 2>&1; module load frameworks/2026.1.0
cd /lus/flare/projects/S2Storm/s2storm/runs/bing/env
C=/lus/flare/projects/S2Storm/s2storm/envs/constraints_frameworks.txt
nice fuxi-xpu/bin/python -m pip install --no-cache-dir -c $C openvino==2025.4.1 && fuxi-xpu/bin/python -c "import openvino as ov; print('ov', ov.__version__)" && echo FUXI_OV_DONE
python -m venv --system-site-packages --without-pip ucs2s-xpu
# same versions as envs/ucs2s-xpu (dist-info names, 2026-11-11), whose files partly sit on the hung OST 9
nice ucs2s-xpu/bin/python -m pip install --no-cache-dir -c $C cartopy==0.26.0 cf_xarray==0.11.3 cftime==1.6.6.1 dask==2026.8.0 \
  netcdf4==1.7.5.1.post1 numcodecs==0.17.1 pyproj==3.8.0 pyshp==3.1.6 ruamel.yaml==0.19.1 shapely==2.2.0 timm==1.0.14 \
  torch_harmonics==0.9.1 wandb==0.31.0 xarray==2026.9.0 zarr==3.4.2 toolz==1.2.0 partd==1.4.2 locket==1.0.0 donfig==0.8.1.post1 \
  google_crc32c==1.9.0 gcsfs h5netcdf
ucs2s-xpu/bin/python -c "import timm, xarray, cartopy, torch_harmonics, cf_xarray, ruamel.yaml, zarr, gcsfs; from timm.models.layers import DropPath; print('uc env ok', timm.__version__, xarray.__version__)" && echo UC_DONE
ucs2s-xpu/bin/python -m pip freeze > ucs2s-xpu.freeze.txt
