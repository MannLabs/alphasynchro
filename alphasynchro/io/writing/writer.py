#!python
'''Module to write spectral data to MS2 output files.'''

# builtin
import dataclasses
import abc

# local
import alphasynchro.ms.peaks.precursors
import alphasynchro.ms.peaks.fragments
import alphasynchro.ms.transitions.frame_transitions


@dataclasses.dataclass
class Writer(abc.ABC):

	file_name: str
	precursors: alphasynchro.ms.peaks.precursors.Precursors
	fragments: alphasynchro.ms.peaks.fragments.Fragments
	transitions: alphasynchro.ms.transitions.frame_transitions.Transitions

	@abc.abstractmethod
	def write_to_file(self) -> None:
		pass
