#!/usr/bin/env bash

sample_path=/mnt/d/raw/test
sample_name=sample_name
out_path=/mnt/d/results/synchro_test
config_path=/mnt/d/configs/synchro_config
species=human
mkdir -p "$out_path"

(
    time (
        time alphatims export hdf "$sample_path"/"$sample_name".d &&
        time timspeak run_pipeline "$config_path"/peakpicker_config.json "$sample_path"/"$sample_name".d "$out_path"/"$sample_name".clusters.hdf &&
        # time alphasynchro create_spectra --analysis_file_name "$out_path"/"$sample_name".spectra.hdf --cluster_file_name "$out_path"/"$sample_name".clusters.hdf &&
        time alphasynchro create_spectra --diapasef --analysis_file_name "$out_path"/"$sample_name".spectra.hdf --cluster_file_name "$out_path"/"$sample_name".clusters.hdf &&
        time alphasynchro write_mgf --analysis_file_name "$out_path"/"$sample_name".spectra.hdf --spectra_file_name "$out_path"/"$sample_name".spectra.mgf &&
        time sage "$config_path"/sage_config.json -f "$config_path"/"$species".fasta "$out_path"/"$sample_name".spectra.mgf 3>&1 1>&2 2>&3 &&
        mv results.sage.tsv "$out_path"/"$sample_name".sage.tsv &&
        mv results.json "$out_path"/"$sample_name".sage.json
    )
) | tee "$out_path"/"$sample_name".log.txt
