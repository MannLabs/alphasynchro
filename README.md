# AlphaSynchro

This package is intended to convert a clusters.hdf file with alphasynchro data into a spectrum file.

## Installation

* Steps
  * Download timspeak: `git clone https://github.com/MannLabs/timspeak.git`
  * Download alphasynchro: `git clone https://github.com/MannLabs/alphasynchro.git`
  * Create conda env: `conda create -n synchro python=3.10 -y`
  * Activate: `conda activate synchro`
  * Install Sage: `conda install -c bioconda -c conda-forge sage-proteomics -y`
  * Install timspeak and alphasynchro: `pip install -e ./timspeak -e ./alphasynchro`

## Usage

* Required
  * Config folder (e.g. [example](example/)) containing:
    * timspeak.config
    * sage.config
    * species.fasta
  * raw_data.d
* Usage
  * Copy-paste [examplary command-line input](example/cmd.sh) in the [example folder](example/) and modify.
  * Note: if working with regular dia-PASEF rather than synchro-PASEF, use line 15 instead of 14 (add the `--diapasef`, but furtherwise identical).
  * Run `./example/cmd.sh`
