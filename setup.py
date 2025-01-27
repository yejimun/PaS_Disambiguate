from setuptools import setup, find_packages

setup(
    name='PaS_CrowdNav2',
    version='0.0.1',
    packages=find_packages(where='./'),  # Automatically find packages in the current directory
    package_dir={'': './'},  # If your packages are in a different directory, specify it here
    python_requires='>=3.8',  # Originally, 3.9 Specify the minimum required Python version
    install_requires=[
        'numpy>=1.19',
        'tqdm>=4.62',
        'matplotlib>=3.5',
        'dill>=0.3.4',
        'pandas>=1.4.1',
        'pyarrow>=7.0.0',
        'torch>=1.13.1',
        'zarr>=2.11.0',
        'kornia>=0.6.4',
        'pathos>=0.2.9',
        'seaborn>=0.12',
        'protobuf>=3.19.4',  # for trajdata map api
        'orjson>=3.5.1',
        'ncls>=0.0.57',
        'wandb',
        'mpc @ git+https://github.com/locuslab/mpc.pytorch.git',
        'trajdata[nusc] @ git+https://github.com/NVlabs/trajdata.git@d714b82dadec80c62e6413ae9a0feb42a517be57'
    ],
    # If there are any package data, scripts, or other resources to include, specify them here
    include_package_data=True,
    # Include any package data files here
    package_data={
        # Specify any package-specific data here
    },
    # Entry points for executable scripts could also be defined here, if necessary
    entry_points={
        # Define console scripts or GUI scripts
    },
)