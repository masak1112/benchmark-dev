python postprocess_precip_forecast_parallel.py \
    --parallel-mode files \
    --max-workers 50 \
    --years 2019 2020 2021 2022 2023 2024 \
    --months 5 6 7 \
    --input-dir "/scratch/10786/bgong1/PanguWeather/v2.0/HPC_scripts/results/S2S/10/predictions" \
    --output-dir "/scratch/10441/rmasiwal/s2s_model_out/regrid_3dayrollout" \
    --grid-file "/scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/cdo_grid_config/2p0_grid_ind.txt" \
    --file-pattern "pangu_plasim_10_24h_45step_{year}*_ens_*.nc" \
    --output-filename "{year}.nc"