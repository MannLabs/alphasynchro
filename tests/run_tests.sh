conda activate alphasynchro_testing
find ../alphasynchro -name '*.py' | xargs wc -l | sort -nr
# . setup_test_clusters.sh
coverage run --source=../alphasynchro -m pytest ./unit_tests
coverage html
coverage report
conda deactivate
