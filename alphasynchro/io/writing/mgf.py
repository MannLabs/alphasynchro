#!python
'''Module to write MGF files from spectral data.'''


# local
import alphasynchro.io.writing.writer


class Writer(alphasynchro.io.writing.writer.Writer):

	def write_to_file(self) -> None:
		with open(self.file_name, "w") as outfile:
			self.write_all_spectra(outfile)

	def write_all_spectra(self, outfile) -> None:
		for index, precursor_index in enumerate(self.transitions.precursor_indices):
			outfile.write("BEGIN IONS\n")
			self.write_spectrum_header(precursor_index, outfile)
			self.write_spectrum_peaks(index, outfile)
			outfile.write("END IONS\n")

	def write_spectrum_header(self, precursor_index: int, outfile) -> None:
		mz = self.precursors.aggregate_data.mz_weighted_average[precursor_index]
		im = self.precursors.aggregate_data.im_weighted_average[precursor_index]
		rt = self.precursors.aggregate_data.rt_weighted_average[precursor_index]
		charge = self.precursors.aggregate_data.charge[precursor_index]
		outfile.write(f"TITLE=Precursor:{precursor_index + 1};IM:{im:.6f}\n")
		outfile.write(f"PEPMASS={mz:.4f}\n")
		outfile.write(f"CHARGE={charge}\n")
		outfile.write(f"RTINSECONDS={rt:.4f}\n")

	def write_spectrum_peaks(self, precursor_index: int, outfile) -> None:
		fragment_indices = self.transitions.get_values(precursor_index)
		for fragment_index in fragment_indices:
			mz = self.fragments.aggregate_data.mz_weighted_average[fragment_index]
			intensity = self.fragments.aggregate_data.summed_intensity[fragment_index]
			outfile.write(f"{mz:.4f} {intensity:.1f}\n")
