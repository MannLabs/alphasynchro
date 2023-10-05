#!python
'''Module to log progress and Python/platform info to stream and file.'''


# builtin
import logging
import os
import sys

# local
import alphasynchro


BASE_PATH = os.path.dirname(__file__)
LOG_PATH = os.path.join(os.path.expanduser('~'), ".logs")


def set_logger(
    *,
    log_file_name="",
    stream: bool = True,
    log_level: int = logging.INFO,
    overwrite: bool = False,
) -> str:
    root = logging.getLogger()
    formatter = _set_formatter()
    root.setLevel(log_level)
    while root.hasHandlers():
        root.removeHandler(root.handlers[0])
    if stream:
        stream_handler = _get_stream_handler(log_level, formatter)
        root.addHandler(stream_handler)
    if log_file_name is not None:
        file_handler = _get_file_handler(
            log_file_name,
            overwrite,
            log_level,
            formatter,
        )
        root.addHandler(file_handler)
    return log_file_name


def _set_formatter():
    return logging.Formatter(
        '%(asctime)s> %(message)s', "%Y-%m-%d %H:%M:%S"
    )


def _get_stream_handler(log_level: int, formatter):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    return stream_handler


def _get_file_handler(
    log_file_name: str,
    overwrite: bool,
    log_level: int,
    formatter,
):
        if log_file_name == "":
            if not os.path.exists(LOG_PATH):
                os.makedirs(LOG_PATH)
            log_file_name = LOG_PATH
        log_file_name = os.path.abspath(log_file_name)
        if os.path.isdir(log_file_name):
            current_time = _get_current_time()
            log_file_name = os.path.join(
                log_file_name,
                f"log_{current_time}.txt"
            )
        directory = os.path.dirname(log_file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if overwrite:
            file_handler = logging.FileHandler(log_file_name, mode="w")
        else:
            file_handler = logging.FileHandler(log_file_name, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        return file_handler


def _get_current_time() -> str:
    import time
    current_time = time.localtime()
    current_time = "".join(
        [
            f'{current_time.tm_year:04}',
            f'{current_time.tm_mon:02}',
            f'{current_time.tm_mday:02}',
            f'{current_time.tm_hour:02}',
            f'{current_time.tm_min:02}',
            f'{current_time.tm_sec:02}',
        ]
    )
    return current_time


def show_platform_info() -> None:
    import platform
    import psutil
    logging.info("Platform information:")
    logging.info(f"system        - {platform.system()}")
    logging.info(f"release       - {platform.release()}")
    if platform.system() == "Darwin":
        logging.info(f"version       - {platform.mac_ver()[0]}")
    else:
        logging.info(f"version       - {platform.version()}")
    logging.info(f"machine       - {platform.machine()}")
    logging.info(f"processor     - {platform.processor()}")
    logging.info(
        f"cpu count     - {psutil.cpu_count()}"
        # f" ({100 - psutil.cpu_percent()}% unused)"
    )
    #logging.info(f"cpu frequency - {psutil.cpu_freq().current:.2f} Mhz")
    logging.info(
        f"ram           - "
        f"{psutil.virtual_memory().available/1024**3:.1f}/"
        f"{psutil.virtual_memory().total/1024**3:.1f} Gb "
        f"(available/total)"
    )
    logging.info("")


def show_python_info() -> None:
    import platform
    module_versions = {
        "python": platform.python_version(),
        "alphasynchro": alphasynchro.__version__
    }
    for (module_name, module_version) in _generate_modules():
        module_versions[module_name] = module_version
    max_len = max(len(key) for key in module_versions)
    logging.info("Python information:")
    for key, value in sorted(module_versions.items()):
        logging.info(f"{key:<{max_len}} - {value}")
    logging.info("")


def _generate_modules():
    import importlib.metadata
    requirements = importlib.metadata.requires("alphasynchro")
    for requirement in requirements:
        parts = requirement.split(";")
        if len(parts) > 1:
            if "development" in parts[1]:
                continue
            if "win32" in parts[1]:
                continue
        module_name = parts[0].split("=")[0].split()[0]
        try:
            module_version = importlib.metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:
            module_version = ""
        yield (module_name, module_version)
