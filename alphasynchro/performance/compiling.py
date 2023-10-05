#!python
'''Module to decorate classes as njit_dataclass.'''


# builtin
import ast
import textwrap
import inspect
import types
import dataclasses

# external
import numba
import pandas as pd
import numpy as np


def njit_dataclass(
    _cls=None,
    kw_only=True,
    frozen=True,
    eq=True,
    **dataclass_kwargs
):
    def wrapper(_cls):
        if eq:
            _cls.__eq__ = __eq__
            _cls.__hash__ = __hash__
        _set_njit_compilation_methods(_cls)
        _set_or_update_post_init(_cls)
        dataclass_decorator = dataclasses.dataclass(
            kw_only=kw_only,
            frozen=frozen,
            eq=eq,
            **dataclass_kwargs
        )
        return dataclass_decorator(_cls)
    if _cls is None:
        return wrapper
    else:
        return wrapper(_cls)


def _set_njit_compilation_methods(_cls) -> None:
    _cls.set_njit_methods = set_njit_methods


def set_njit_methods(self) -> None:
    if not is_regular_object_with_dict(self):
        return
    create_njit_module_for_object(self)
    for func in iterate_over_callables_from_object(self):
        try:
            tree = get_source_tree_from_callable(func)
        except OSError:
            pass
        else:
            if tree_has_decorator_containing_name(tree, "njit"):
                add_njit_function_to_object_njit_module(self, func, tree)
                overwrite_object_function_with_njit_function(self, func)


def is_regular_object_with_dict(self) -> bool:
    is_regular = not isinstance(self, numba.core.registry.CPUDispatcher)
    has_dict = hasattr(self, "__dict__")
    return has_dict and is_regular


def create_njit_module_for_object(self) -> None:
    module = create_module(self)
    # TODO include line below in unittests
    module.__dict__.update({"np": np, "numba": numba})
    module.__dict__.update(inspect.getmodule(self).__dict__)
    module.self = module
    object.__setattr__(self, "__njit__", module)


def create_module(x):
    if has_njit_module(x):
        return x.__njit__
    if is_pandas_dataframe(x):
        return create_module_from_dataframe(x)
    elif is_regular_object_with_dict(x):
        return create_module_from_object(x)
    else:
        return x


def is_module(x) -> bool:
    return hasattr(x, "__njit__")


def has_njit_module(x) -> bool:
    return isinstance(x, types.ModuleType)


def is_pandas_dataframe(x) -> bool:
    return isinstance(x, pd.DataFrame)


def create_module_from_dataframe(df):
    module = types.ModuleType(f"{df.__class__}_{id(df)}")
    for column in df.columns:
        module.__dict__[column] = np.array(df[column].values, copy=False)
    for key, value in df.__dict__.items():
        if not key.startswith("_"):
            module.__dict__[key] = create_module(value)
    return module


def create_module_from_object(self):
    module = types.ModuleType(f"{self.__class__}_{id(self)}")
    for key, value in self.__dict__.items():
        if not key.startswith("__"):
            module.__dict__[key] = create_module(value)
    return module


def iterate_over_callables_from_object(self) -> (callable):
    for key in list(dir(self)):
        if key.startswith("__"):
            continue
        potential_func = eval(f"self.{key}")
        if callable(potential_func):
            yield potential_func


def get_source_tree_from_callable(func: callable):
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    return ast.parse(src)


def tree_has_decorator_containing_name(tree, name) -> bool:
    for decorator in tree.body[0].decorator_list:
        if decorator_contains(decorator, name):
            return True
    return False


def decorator_contains(decorator, name: str) -> bool:
    # TODO: Should be properly parsed!
    if name in ast.unparse(decorator):
        return True
    return False


def add_njit_function_to_object_njit_module(self, func, tree) -> None:
    src = create_src_without_self_and_decorators_from_function_tree(tree)
    exec(src, self.__njit__.__dict__)
    nogil = tree_has_decorator_containing_name(tree, "nogil")
    func_ = numba.njit(nogil=nogil)(self.__njit__.__dict__[func.__name__])
    src = f"object.__setattr__(self.__njit__, '{func.__name__}', func_)"
    exec(src)


def create_src_without_self_and_decorators_from_function_tree(tree) -> str:
    # TODO: removes the nogil decorator as well!
    origonal_decorators = tree.body[0].decorator_list
    origonal_args = tree.body[0].args.args
    tree.body[0].decorator_list = []
    tree.body[0].args.args = tree.body[0].args.args[1:]
    src = ast.unparse(tree)
    tree.body[0].decorator_list = origonal_decorators
    tree.body[0].args.args = origonal_args
    return src


def overwrite_object_function_with_njit_function(self, func: callable) -> None:
    src = f"object.__setattr__(self, '{func.__name__}', self.__njit__.{func.__name__})"
    exec(src)


def njit(*args, **kwargs):
    return numba.njit(*args, **kwargs)


def _set_or_update_post_init(_cls) -> None:
    for superclass in _superclass_generator(_cls):
        if not _has_post_init(superclass):
            _set_blank_post_init(superclass)
        if not _has_original_post_init(superclass):
            _update_post_init(superclass)


def _superclass_generator(_cls):
    for superclass in _cls.__mro__[:-1][::-1]:
        yield superclass


def _has_post_init(_cls) -> bool:
    return hasattr(_cls, "__post_init__")


def _set_blank_post_init(_cls) -> None:
    _cls.__post_init__ = __blank_post_init__


def __blank_post_init__(self):
    superclass = _determine_superclass(self)
    if _has_post_init(superclass):
        superclass.__post_init__()


def _determine_superclass(self):
    return super(
        self.__class__.__mro__[self.__class_index__],
        self
    )


def _has_original_post_init(_cls) -> bool:
    return hasattr(_cls, "__original_post_init__")


def _update_post_init(superclass) -> None:
    superclass.__original_post_init__ = superclass.__post_init__
    superclass.__post_init__ = __post_init__


def __post_init__(self):
    if not hasattr(self, "__class_index__"):
        object.__setattr__(self, "__class_index__", -1)
    object.__setattr__(self, "__class_index__", self.__class_index__ + 1)
    self.__class__.__mro__[self.__class_index__].__original_post_init__(self)
    object.__setattr__(self, "__class_index__", self.__class_index__ - 1)
    if _is_root_class(self):
        _update_root_class(self)


def _is_root_class(self) -> bool:
    return self.__class_index__ == -1


def _update_root_class(self):
    if not hasattr(self, "__njit__"):
        self.set_njit_methods()
    del self.__dict__["__class_index__"]


def __eq__(self, other):
    return hash(self) == hash(other)


def __hash__(self):
    if not hasattr(self, "__hash_value__"):
        hash_list = [hash(self.__class__)]
        for key, value_type in self.__annotations__.items():
            value = self.__getattribute__(key)
            if isinstance(value_type, types.GenericAlias):
                value_type = type(value)
            if value_type is np.ndarray:
                hash_value = hash_array(value)
            else:
                hash_value = hash(value)
            hash_list.append(hash_value)
        __hash_value__ = abs(hash(tuple(hash_list)))
        object.__setattr__(self, "__hash_value__", __hash_value__)
    return self.__hash_value__


@numba.njit()
def hash_array(
    arr: np.ndarray,
    offset: int = 43
) -> int:
    hash_value = offset
    byte_array = np.frombuffer(arr, dtype=np.int8)
    for i, byte_val in enumerate(byte_array, offset):
        hash_value ^= (byte_val*offset) << (i%53)
    return hash_value
