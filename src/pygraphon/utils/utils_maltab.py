import matlab 

def npArray2Matlab(x):
    return matlab.double(x.tolist())

def setupMatlabEngine(eng, paths):
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