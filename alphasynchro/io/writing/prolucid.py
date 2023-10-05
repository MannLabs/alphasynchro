#!python
'''Module to write Prolucid files from spectral data.'''


# local
import alphasynchro.io.writing.writer


PROTONMASS = 1.007276466621


class Writer(alphasynchro.io.writing.writer.Writer):

	def write_to_file(self) -> None:
		with open(self.file_name, "w") as outfile:
			self.write_header(outfile)
			self.write_all_spectra(outfile)

	def write_header(self, outfile) -> None:
		import datetime
		outfile.write(
			f"H\tCreation Date\t{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
		)

	def write_all_spectra(self, outfile) -> None:
		for index, precursor_index in enumerate(self.transitions.precursor_indices):
			self.write_spectrum_header(precursor_index, outfile)
			self.write_spectrum_peaks(index, outfile)

	def write_spectrum_header(self, precursor_index, outfile):
		mz = self.precursors.aggregate_data.mz_weighted_average[precursor_index]
		charge = self.precursors.aggregate_data.charge[precursor_index]
		mhp = (mz - PROTONMASS) * charge + PROTONMASS
		rt = self.precursors.aggregate_data.rt_weighted_average[precursor_index]
		im = self.precursors.aggregate_data.im_weighted_average[precursor_index]
		charge = self.precursors.aggregate_data.charge[precursor_index]
		outfile.write(f"S\t{precursor_index + 1}\t{precursor_index + 1}\t{mz:.4f}\n")
		outfile.write(f"I\tTIMSTOF_Precursor_ID\t{precursor_index + 1}\n")
		outfile.write(f"I\tRetTime\t{rt:.4f}\n")
		outfile.write(f"I\tIon Mobility\t{im:.6f}\n")
		outfile.write(f"Z\t{charge}\t{mhp:.4f}\n")


	def write_spectrum_peaks(self, precursor_index, outfile):
		fragment_indices = self.transitions.get_values(precursor_index)
		for fragment_index in fragment_indices:
			mz = self.fragments.aggregate_data.mz_weighted_average[fragment_index]
			intensity = self.fragments.aggregate_data.summed_intensity[fragment_index]
			outfile.write(f"{mz:.4f} {intensity:.1f}")
