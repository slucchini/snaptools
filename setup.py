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
    # install_requires=[
    #     "matplotlib >= 3.3.4",
    #     "numpy >= 1.20.1",
    #     "astropy >= 4.2",
    #     "h5py >= 2.8.0"
    # ]
)