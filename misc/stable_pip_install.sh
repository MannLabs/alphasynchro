conda create -n alphasynchro_testing python=3.10 -y
conda activate alphasynchro_testing
pip install .[stable,development-stable]
alphasynchro
conda deactivate
