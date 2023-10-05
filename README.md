# AlphaSynchro

This package is intended to convert a clusters.hdf file with alphasynchro data into a spectrum file.

## Installation

* Required
  * timspeak.zip
  * alphasynchro.zip

* Steps
  * Unpack timspeak.zip and alphasynchro.zip
  * Create conda env: `conda create -n synchro python=3.10 -y`
  * Activate: `conda activate synchro`
  * Install msconvert (mgf->mzml): `conda install -c bioconda proteowizard -y`
  * Install Sage: `conda install -c bioconda -c conda-forge sage-proteomics -y`
  * Install timspeak and alphasynchro: `pip install -e ./timspeak -e ./alphasynchro`

## Usage

* Required
  * timspeak.config
  * sage.config
  * sample.fasta
  * raw_data.d

See the [command-line input](example/cmd.txt) in the [example folder](example/).
