#!python
'''Module to write HDF files from spectral data.'''


# local
import alphasynchro.io.writing.writer
import alphasynchro.io.hdf


class Writer(alphasynchro.io.writing.writer.Writer):

	def write_to_file(self) -> None:
		outfile = alphasynchro.io.hdf.HDFObject.from_file(
			self.file_name,
			new=True
		)
		self.write_precursors(outfile)
		self.write_fragments(outfile)

	def write_precursors(self, outfile) -> None:
		precursors = outfile.set_group("precursors")
		for name, array in [
			("mz", self.precursors.aggregate_data.mz_weighted_average),
			("im", self.precursors.aggregate_data.im_weighted_average),
			("rt", self.precursors.aggregate_data.rt_weighted_average),
			("charge", self.precursors.aggregate_data.charge),
			("fragment_start_index", self.transitions.indptr[:-1]),
			("fragment_end_index", self.transitions.indptr[1:]),
		]:
			precursors.set_mmap(name, array[self.transitions.precursor_indices])

	def write_fragments(self, outfile) -> None:
		fragments = outfile.set_group("fragments")
		for name, array in [
			("mz", self.fragments.aggregate_data.mz_weighted_average),
			("im", self.fragments.aggregate_data.im_weighted_average),
			("rt", self.fragments.aggregate_data.rt_weighted_average),
			("intensity", self.fragments.aggregate_data.summed_intensity),
		]:
			fragments.set_mmap(name, array[self.transitions.values])
