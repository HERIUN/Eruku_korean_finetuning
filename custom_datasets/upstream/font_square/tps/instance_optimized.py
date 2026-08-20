import numpy

from . import functions_optimized as functions

__all__ = ['TPS', 'TPSOptimized']


class TPS:
    """The thin plate spline deformation warpping.
    """

    def __init__(self,
                 control_points: numpy.ndarray,
                 target_points: numpy.ndarray,
                 lambda_: float = 0.,
                 solver: str = 'exact'):
        """Create a instance that preserve the TPS coefficients.

        Arguments
        ---------
            control_points : numpy.array
                p by d vector of control points
            target_points : numpy.array
                p by d vector of corresponding target points on the deformed
                surface
            lambda_ : float
                regularization parameter
            solver : str
                the solver to get the coefficients. default is 'exact' for the
                exact solution. Or use 'lstsq' for the least square solution.
        """
        self.control_points = control_points
        self.coefficient = functions.find_coefficients(
            control_points, target_points, lambda_, solver)

    def __call__(self, source_points):
        """Transform the source points form the original surface to the
        destination (deformed) surface.

        Arguments
        ---------
            source_points : numpy.array
                n by d array of source points to be transformed
        """
        return functions.transform(source_points, self.control_points,
                                   self.coefficient)

    transform = __call__


class TPSOptimized(TPS):
    """Optimized version of TPS that uses C++ backend when available.
    
    This class provides the same interface as TPS but automatically uses
    the C++ optimized implementation when available, with fallback to 
    the numpy implementation.
    """
    
    def __init__(self,
                 control_points: numpy.ndarray,
                 target_points: numpy.ndarray,
                 lambda_: float = 0.,
                 solver: str = 'exact',
                 verbose: bool = False):
        """Create an optimized TPS instance.

        Arguments
        ---------
            control_points : numpy.array
                p by d vector of control points
            target_points : numpy.array
                p by d vector of corresponding target points on the deformed
                surface
            lambda_ : float
                regularization parameter
            solver : str
                the solver to get the coefficients. default is 'exact' for the
                exact solution. Or use 'lstsq' for the least square solution.
            verbose : bool
                whether to print information about which backend is being used
        """
        self.verbose = verbose
        if verbose and functions.is_cpp_available():
            print("Using C++ optimized TPS backend")
        elif verbose:
            print("Using numpy TPS backend")
            
        super().__init__(control_points, target_points, lambda_, solver)
    
    @classmethod
    def benchmark(cls, 
                  control_points: numpy.ndarray, 
                  target_points: numpy.ndarray,
                  source_points: numpy.ndarray,
                  iterations: int = 10) -> dict:
        """Benchmark the performance of both implementations.
        
        Arguments
        ---------
            control_points : numpy.array
                p by d vector of control points
            target_points : numpy.array
                p by d vector of target points
            source_points : numpy.array
                n by d array of source points to transform
            iterations : int
                number of iterations for benchmarking
                
        Returns
        -------
            dict : benchmark results including timing and speedup information
        """
        return functions.benchmark_performance(
            control_points, target_points, source_points, iterations
        )
    
    @staticmethod
    def is_optimized() -> bool:
        """Check if the C++ optimized backend is available."""
        return functions.is_cpp_available() 