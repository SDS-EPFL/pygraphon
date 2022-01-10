import matlab
import matlab.engine
import pathlib
import os


def npArray2Matlab(x):
    """Convert a numpy array to a matlab array.

    """

    return matlab.double(x.tolist())


def setupMatlabEngine(eng: matlab.engine.MatlabEngine, paths: str):
    """
    Setup the matlab engine to use the correct paths to the matlab scripts."""
    if eng is None:
        if paths is None:
            raise ValueError(
                "no path to network histogram approximation files in matlab, "
                + "please provide either matlab engine or path to scripts"
            )
        elif eng is None:
            eng = matlab.engine.start_matlab()

        eng.addpath(paths, nargout=0)
    return eng


def getMatlabPaths(function_name="nethist"):
    """Dirty trick to get the correct paths of the matlab scripts:

    ### Any change in name or structure of the code directory will make this fail !

    Returns:
        [str]: paths to matlab file containing function_name.m
    """
    pathFile = pathlib.Path(__file__).parent.parent.absolute()
    path = os.path.join(pathFile, "matlab_functions", f"{function_name}.m")
    return path
