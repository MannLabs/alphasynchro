conda activate alphasynchro_testing
python -c "import setup_test_clusters as setup; setup.create_test_cluster_hdf_object()"
conda deactivate
