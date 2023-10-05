#!python
'''Module to store multiple arrays of equal length as a njit_dataclass.'''


# external
import pandas as pd

# local
import alphasynchro.performance.compiling


@alphasynchro.performance.compiling.njit_dataclass
class DataFrame:

    shape: tuple[int,int]

    def __init__(self, **named_arrays):
        if "shape" in named_arrays:
            named_arrays.pop("shape")
        if "columns" in named_arrays:
            named_arrays.pop("columns")
        for name, array in named_arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "columns", list(named_arrays.keys()))
        object.__setattr__(self, "shape", (len(array), len(named_arrays)))

    def as_dataframe(self, indices=..., *, columns=None) -> pd.DataFrame:
        if columns is None:
            columns = self.columns
        return pd.DataFrame(
            {
                name: self.__getattribute__(name)[indices] for name in columns
            }
        )

    def __len__(self):
        return self.shape[0]
