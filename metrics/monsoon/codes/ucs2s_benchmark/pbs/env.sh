# Common environment for the UC-S2S monsoon benchmark PBS jobs.
# Jobs must be submitted from the package directory (submit_all.sh does this), so that
# PBS_O_WORKDIR is the package and relative "#PBS -o logs/..." paths resolve there.
PKG=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-/glade/work/rajatm/conda-envs/ngcm_env/bin/python}     # any env with xarray, geopandas, seaborn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PKG"
