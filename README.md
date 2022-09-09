#SnapTools

Bespoke tools for working with isolated galaxy snapshots produced by Gadget and GIZMO. Based on code developed by Steven Pardy.

#Installation

SnapTools requires several common python packages (tested versions in parentheses):

- astropy (5.0.4)
- h5py (3.6.0)
- multiprocess (0.70.12.2)
- matplotlib (3.5.1)
- numpy (1.20.1)

SnapTools also uses the pNbody package available here: [http://obswww.unige.ch/~revaz/pNbody/](http://obswww.unige.ch/~revaz/pNbody/).

Also included are some utility functions for use with the pygad package: [https://bitbucket.org/broett/pygad/src/master/](https://bitbucket.org/broett/pygad/src/master/). Without pygad installed all functionality outside of the pygad_utils file will work.

## Local pip installation

The recommended installation method for SnapTools is to clone the git repo to your computer, and then use pip to install the package locally.

```bash
$> git clone git@github.com:slucchini/snaptools.git
$> python3 -m pip install -e ./snaptools/
```

The `-e` argument allows for live updating of the package upon modification of the source. You can leave it out if you don't plan to edit the source python.