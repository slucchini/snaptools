import setuptools

setuptools.setup(
    name="snaptools",
    version="0.1",
    author="Scott Lucchini",
    author_email="scott.lucchini@gmail.com",
    description="Tools for working with GIZMO snapshots",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "astropy >= 5.0",
        "h5py >= 3.6.0",
        "multiprocess >= 0.70.12.2",
        "matplotlib >= 3.5.1",
        "numpy >= 1.20.1",
    ]
)