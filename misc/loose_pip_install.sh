conda create -n alphasynchro_testing python=3.10 -y
conda activate alphasynchro_testing
pip install .[development]
alphasynchro
conda deactivate
