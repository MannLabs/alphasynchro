#!python

# builtin
import setuptools
import re
import os
import datetime

# local
import alphasynchro


def get_version():
    version = alphasynchro.__version__
    return version
    # TODO fails and is unclear to me
    stage = os.environ.get("BRANCH", "local")
    build_number = datetime.now().strftime('%Y%m%d%H%M%S')
    addition = f'.{build_number}' if stage == 'main' else f'a{build_number}'
    if stage in ['develop', 'dev']:
        addition = f'b{build_number}'
    if stage.startswith('release'):
        addition = f'rc{build_number}'
    return f"{version}{addition}"


def get_long_description():
    with open("README.md", "r") as readme_file:
        long_description = readme_file.read()
    return long_description


def get_requirement_files():
    requirement_file_names = {}
    root_path = os.path.dirname(__file__)
    requirements_path = os.path.join(root_path, "requirements")
    for file_name in os.listdir(requirements_path):
        full_file_name = os.path.join(requirements_path, file_name)
        if os.path.isfile(full_file_name):
            file_name_without_extension = os.path.splitext(file_name)[0]
            extra = "_".join(file_name_without_extension.split("_")[1:])
            requirement_file_names[extra] = full_file_name
    return requirement_file_names


def get_requirements():
    extra_requirements = {}
    requirement_file_names = get_requirement_files()
    for extra, requirement_file_name in requirement_file_names.items():
        with open(requirement_file_name) as requirements_file:
            if extra == "release":
                extra_requirements["release"] = requirements_file.readlines()
            else:
                extra_stable = f"{extra}-stable" if extra != "" else "stable"
                extra_requirements[extra_stable] = []
                extra_requirements[extra] = []
                for line in requirements_file:
                    extra_requirements[extra_stable].append(line)
                    requirement, *version = re.split("[><=~!]", line)
                    requirement == requirement.strip()
                    extra_requirements[extra].append(requirement)
    base_requirements = extra_requirements.pop("")
    return base_requirements, extra_requirements


def create_pip_wheel():
    base_requirements, extra_requirements = get_requirements()
    version = get_version()
    setuptools.setup(
        name="alphasynchro",
        version=version,
        license=alphasynchro.__license__,
        description=alphasynchro.__description__,
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author=alphasynchro.__author__,
        author_email=alphasynchro.__author_email__,
        url=alphasynchro.__github__,
        project_urls=alphasynchro.__urls__,
        keywords=alphasynchro.__keywords__,
        classifiers=alphasynchro.__classifiers__,
        packages=setuptools.find_namespace_packages(include=["alphasynchro.*", "alphasynchro"]),
        include_package_data=True,
        # package_data={name: ['*.avro']}, # TODO not sure how this sorks, I always got data through a MANIFEST.in file
        entry_points={
            "console_scripts": alphasynchro.__console_scripts__,
        },
        install_requires=base_requirements,
        extras_require=extra_requirements,
        python_requires=alphasynchro.__python_version__,
    )


if __name__ == "__main__":
    create_pip_wheel()
