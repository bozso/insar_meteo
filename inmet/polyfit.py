import numpy as np
from pickle import dump, load


class PolyFit(object):

    @staticmethod
    def make_jacobi(x, deg):
        assert deg >= 1
        
        return np.vander(x, deg + 1)

        
    @staticmethod
    def polyfit(x, y, jacobi=None, deg=None):
        assert jacobi is not None and deg is not None, "design, deg"
        
        if design is None:
            jacobi = PolyFit.make_jacobi(x, deg)

        # coeffs[0]: polynom coeffcients are in the columns
        # coeffs[1]: residuals
        # coeffs[2]: rank of design matrix
        # coeffs[3]: singular values of design matrix
        return np.linalg.lstsq(design, y)[0]

    
    def __init__(self, x, y, deg, order="c"):
        order = order.upper()
        
        x, y, _deg = \
        np.array(x, order=order), np.array(y, order=order), np.array(deg)
        
        mdeg = _deg.max()
        
        jacobi = PolyFit.make_jacobi(x, mdeg)
        
        if y.ndim > 1:
            self.nfit = y.shape[0 if order == "C" else 1]
            self.deg = _deg
            
            itr = np.nditer(y, op_flags=["readonly"], order=order)
            
            coeffs = (PolyFit.polyfit(x, Y, deg=_deg[ii], jacobi[:ii,:])
                      ii, Y for enumerate(itr))
            
            self.coeffs = np.array(tuple(coeffs))
        else:
            self.nfit = 1
            self.deg = mdeg
            self.coeffs = PolyFit.polyfit(x, y, jacobi=jacobi)    
    

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            self = load(f)
        
    
    def save(self, path):
        with open(path, "wb") as f:
            dump(self, f)
    
        
