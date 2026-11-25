# Task
Calculate MJO (Madden-Julian Oscillation) scores for probabilistic forecasts using an existing pipeline and generate phase space plots.

## Input Data
Forecast data location:
/glade/derecho/scratch/bgong/fuxi_s2s
Let's only use the year from 2017-2021 for calculating the MJO

## Reference Pipeline
Use the existing MJO evaluation pipeline (do not rewrite from scratch):
/glade/work/bgong/MJO/chi250_eval

## Environment
Use the following conda environment for all dependencies:
/glade/work/bgong/mjocast_env

## Output
Generate phase space plots matching the format and style of this reference output:
/glade/work/bgong/MJO/chi250_eval/plots_chi200/obs_phase_space_20180627.png

Also generate the bivariate ACC as /glade/work/bgong/MJO/chi250_eval/plots_chi200/chi200_bivariate_acc.png


Save all output plots to the same directory:
/glade/work/bgong/MJO/chi250_eval/plots_chi200/

