import numpy
import warnings

# Try to import the C++ optimized version
try:
    from . import tps_cpp  # Relative import for package usage
    _CPP_AVAILABLE = True
    print("TPS C++ optimized backend loaded successfully")
except ImportError:
    try:
        import tps_cpp  # Direct import for standalone usage
        _CPP_AVAILABLE = True
        print("TPS C++ optimized backend loaded successfully")
    except ImportError:
        _CPP_AVAILABLE = False
        warnings.warn("TPS C++ backend not available, falling back to numpy implementation")

__all__ = ['find_coefficients', 'transform']


def cdist(K: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray:
    """Calculate Euclidean distance between K[i, :] and B[j, :].

    Arguments
    ---------
        K : numpy.array
        B : numpy.array
    """
    K = numpy.atleast_2d(K)
    B = numpy.atleast_2d(B)
    assert K.ndim == 2
    assert B.ndim == 2

    K = numpy.expand_dims(K, 1)
    B = numpy.expand_dims(B, 0)
    D = K - B
    return numpy.linalg.norm(D, axis=2)


def pairwise_radial_basis(K: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray:
    """Compute the TPS radial basis function phi(r) between every row-pair of K
    and B where r is the Euclidean distance.

    Arguments
    ---------
        K : numpy.array
            n by d vector containing n d-dimensional points.
        B : numpy.array
            m by d vector containing m d-dimensional points.

    Return
    ------
        P : numpy.array
            n by m matrix where.
            P(i, j) = phi( norm( K(i,:) - B(j,:) ) ),
            where phi(r) = r^2*log(r), if r >= 1
                           r*log(r^r), if r <  1
    """
    if _CPP_AVAILABLE:
        return tps_cpp.TPSOptimized.pairwise_radial_basis(K, B)
    
    # Fallback to original numpy implementation
    # r_mat(i, j) is the Euclidean distance between K(i, :) and B(j, :).
    r_mat = cdist(K, B)

    pwise_cond_ind1 = r_mat >= 1
    pwise_cond_ind2 = r_mat < 1
    r_mat_p1 = r_mat[pwise_cond_ind1]
    r_mat_p2 = r_mat[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = numpy.empty(r_mat.shape)
    P[pwise_cond_ind1] = (r_mat_p1**2) * numpy.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * numpy.log(numpy.power(r_mat_p2, r_mat_p2))

    return P


def find_coefficients(control_points: numpy.ndarray,
                      target_points: numpy.ndarray,
                      lambda_: float = 0.,
                      solver: str = 'exact') -> numpy.ndarray:
    """Given a set of control points and their corresponding points, compute the
    coefficients of the TPS interpolant deforming surface.

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
            the solver to get the coefficients. default is 'exact' for the exact
            solution. Or use 'lstsq' for the least square solution.

    Return
    ------
        coef : numpy.ndarray
            the coefficients

    .. seealso::

        http://cseweb.ucsd.edu/~sjb/pami_tps.pdf
    """
    if _CPP_AVAILABLE:
        return tps_cpp.TPSOptimized.find_coefficients(control_points, target_points, lambda_, solver)
    
    # Fallback to original numpy implementation
    # ensure data type and shape
    control_points = numpy.atleast_2d(control_points)
    target_points = numpy.atleast_2d(target_points)
    if control_points.shape != target_points.shape:
        raise ValueError(
            'Shape of and control points {cp} and target points {tp} are not the same.'.
            format(cp=control_points.shape, tp=target_points.shape))

    p, d = control_points.shape

    # The matrix
    K = pairwise_radial_basis(control_points, control_points)
    P = numpy.hstack([numpy.ones((p, 1)), control_points])

    # Relax the exact interpolation requirement by means of regularization.
    K = K + lambda_ * numpy.identity(p)

    # Target points
    M = numpy.vstack([
        numpy.hstack([K, P]),
        numpy.hstack([P.T, numpy.zeros((d + 1, d + 1))])
    ])
    Y = numpy.vstack([target_points, numpy.zeros((d + 1, d))])

    # solve for M*X = Y.
    # At least d+1 control points should not be in a subspace; e.g. for d=2, at
    # least 3 points are not on a straight line. Otherwise M will be singular.
    solver = solver.lower()
    if solver == 'exact':
        X = numpy.linalg.solve(M, Y)
    elif solver == 'lstsq':
        X, _, _, _ = numpy.linalg.lstsq(M, Y, None)
    else:
        raise ValueError('Unknown solver: ' + solver)

    return X


def transform(source_points: numpy.ndarray, control_points: numpy.ndarray,
              coefficient: numpy.ndarray) -> numpy.ndarray:
    """Transform the source points form the original surface to the destination
    (deformed) surface.

    Arguments
    ---------
        source_points : numpy.array
            n by d array of source points to be transformed
        control_points : numpy.array
            the control points used in the function `find_coefficients`
        coefficient : numpy.array
            the computed coefficients

    Return
    ------
        deformed_points : numpy.array
            n by d array of the transformed point on the target surface
    """
    if _CPP_AVAILABLE:
        return tps_cpp.TPSOptimized.transform(source_points, control_points, coefficient)
    
    # Fallback to original numpy implementation
    source_points = numpy.atleast_2d(source_points)
    control_points = numpy.atleast_2d(control_points)
    if source_points.shape[-1] != control_points.shape[-1]:
        raise ValueError(
            'Dimension of source points ({sd}D) and control points ({cd}D) are not the same.'.
            format(sd=source_points.shape[-1], cd=control_points.shape[-1]))

    n = source_points.shape[0]

    A = pairwise_radial_basis(source_points, control_points)
    K = numpy.hstack([A, numpy.ones((n, 1)), source_points])

    deformed_points = numpy.dot(K, coefficient)
    return deformed_points


# Convenience function to check if C++ backend is available
def is_cpp_available() -> bool:
    """Check if the C++ optimized backend is available."""
    return _CPP_AVAILABLE


# Convenience function to benchmark performance
def benchmark_performance(control_points: numpy.ndarray, 
                        target_points: numpy.ndarray,
                        source_points: numpy.ndarray,
                        iterations: int = 10) -> dict:
    """Benchmark the performance of both implementations.
    
    Returns a dictionary with timing results.
    """
    import time
    global _CPP_AVAILABLE
    
    results = {
        'cpp_available': _CPP_AVAILABLE,
        'iterations': iterations
    }
    
    cpp_time = None
    if _CPP_AVAILABLE:
        # Benchmark C++ implementation
        start_time = time.time()
        for _ in range(iterations):
            coeff = tps_cpp.TPSOptimized.find_coefficients(control_points, target_points)
            _ = tps_cpp.TPSOptimized.transform(source_points, control_points, coeff)
        cpp_time = time.time() - start_time
        results['cpp_time'] = cpp_time
        results['cpp_avg_time'] = cpp_time / iterations
    
    # Benchmark numpy implementation (with C++ disabled)
    original_cpp_state = _CPP_AVAILABLE
    _CPP_AVAILABLE = False
    
    start_time = time.time()
    for _ in range(iterations):
        coeff = find_coefficients(control_points, target_points)
        _ = transform(source_points, control_points, coeff)
    numpy_time = time.time() - start_time
    
    # Restore original state
    _CPP_AVAILABLE = original_cpp_state
    
    results['numpy_time'] = numpy_time
    results['numpy_avg_time'] = numpy_time / iterations
    
    if cpp_time is not None:
        results['speedup'] = numpy_time / cpp_time
    
    return results 