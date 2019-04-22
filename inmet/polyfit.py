from inmet import Save, iteraxis
import numpy as np

__all__ = ["PolyFit"]


class PolyFit(Save):

    @staticmethod
    def make_jacobi(x, deg):
        assert deg >= 1, "deg should be at least 1."
        
        return np.vander(x, deg + 1)

        
    @staticmethod
    def polyfit(x, y, jacobi=None, deg=None):
        assert jacobi is not None or deg is not None, "design, deg"
        
        if jacobi is None:
            jacobi = PolyFit.make_jacobi(x, deg)

        # coeffs[0]: polynom coeffcients are in the columns
        # coeffs[1]: residuals
        # coeffs[2]: rank of design matrix
        # coeffs[3]: singular values of design matrix
        return np.linalg.lstsq(jacobi, y, rcond=None)[0]

    
    def __init__(self, x, y, deg, order="cols"):
        assert order in ("cols", "rows")
        
        axis = 0 if order == "rows" else 1
        
        x, y, _deg = np.array(x), np.array(y), np.array(deg)
        
        mdeg = _deg.max()
        
        jacobi = PolyFit.make_jacobi(x, mdeg)
        
        
        if y.ndim > 1:
            self.nfit = y.shape[axis]
            self.deg = _deg
            self.ncoeffs = _deg + 1
            
            if _deg.size == 1:
                self.coeffs = PolyFit.polyfit(x, y, jacobi=jacobi)
            else:
                coeffs = (PolyFit.polyfit(x, Y, jacobi=jacobi[:,-deg[ii] - 1:])
                          for ii, Y in enumerate(iteraxis(y, axis)))
                
                
                self.coeffs = np.hstack(coeffs)
        else:
            self.nfit = 1
            self.deg = mdeg
            self.coeffs = PolyFit.polyfit(x, y, jacobi=jacobi)    

    
    def __call__(self, x, tensor=True):
        if isinstance(x, (tuple, list)):
            x = np.asarray(x)
        if isinstance(x, np.ndarray) and tensor:
            c = self.coeffs.reshape(self.coeffs.shape + (1,) * x.ndim)
        
        
        if self.nfit == 1:
            c0 = c[0] + x * 0
            
            for coeff in c[1:]:
                c0 = ceoff + c0 * x

            return c0
        else:
            pass