#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>
#include <algorithm>
#include <execution>
#include <vector>

namespace py = pybind11;
using namespace Eigen;

class TPSOptimized {
private:
    static constexpr double EPS = 1e-12;
    
    // Optimized distance calculation using Eigen
    static MatrixXd compute_distances(const MatrixXd& K, const MatrixXd& B) {
        const int n = K.rows();
        const int m = B.rows();
        const int d = K.cols();
        
        MatrixXd distances(n, m);
        
        // Vectorized distance computation
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                distances(i, j) = (K.row(i) - B.row(j)).norm();
            }
        }
        
        return distances;
    }
    
    // Optimized radial basis function computation
    static MatrixXd compute_radial_basis(const MatrixXd& distances) {
        MatrixXd P = MatrixXd::Zero(distances.rows(), distances.cols());
        
        for (int i = 0; i < distances.rows(); ++i) {
            for (int j = 0; j < distances.cols(); ++j) {
                double r = distances(i, j);
                if (r < EPS) {
                    P(i, j) = 0.0;
                } else if (r >= 1.0) {
                    P(i, j) = r * r * std::log(r);
                } else {
                    P(i, j) = r * std::log(std::pow(r, r));
                }
            }
        }
        
        return P;
    }
    
public:
    // Optimized pairwise radial basis function
    static MatrixXd pairwise_radial_basis(const MatrixXd& K, const MatrixXd& B) {
        MatrixXd distances = compute_distances(K, B);
        return compute_radial_basis(distances);
    }
    
    // Optimized coefficient finding with better numerical stability
    static MatrixXd find_coefficients(const MatrixXd& control_points,
                                    const MatrixXd& target_points,
                                    double lambda = 0.0,
                                    const std::string& solver = "exact") {
        
        if (control_points.rows() != target_points.rows() || 
            control_points.cols() != target_points.cols()) {
            throw std::invalid_argument("Control points and target points must have the same dimensions");
        }
        
        const int p = control_points.rows();
        const int d = control_points.cols();
        
        // Compute K matrix (radial basis functions between control points)
        MatrixXd K = pairwise_radial_basis(control_points, control_points);
        
        // Add regularization
        K += lambda * MatrixXd::Identity(p, p);
        
        // Create P matrix [1, control_points]
        MatrixXd P(p, d + 1);
        P.col(0) = VectorXd::Ones(p);
        P.rightCols(d) = control_points;
        
        // Construct the full system matrix M
        MatrixXd M(p + d + 1, p + d + 1);
        M.topLeftCorner(p, p) = K;
        M.topRightCorner(p, d + 1) = P;
        M.bottomLeftCorner(d + 1, p) = P.transpose();
        M.bottomRightCorner(d + 1, d + 1) = MatrixXd::Zero(d + 1, d + 1);
        
        // Construct the target vector Y
        MatrixXd Y(p + d + 1, d);
        Y.topRows(p) = target_points;
        Y.bottomRows(d + 1) = MatrixXd::Zero(d + 1, d);
        
        // Solve the linear system
        MatrixXd X;
        if (solver == "exact") {
            // Use LU decomposition for better numerical stability
            FullPivLU<MatrixXd> lu(M);
            if (!lu.isInvertible()) {
                throw std::runtime_error("Matrix M is singular. Check that control points are not collinear.");
            }
            X = lu.solve(Y);
        } else if (solver == "lstsq") {
            // Use SVD for least squares solution
            X = M.bdcSvd(ComputeThinU | ComputeThinV).solve(Y);
        } else {
            throw std::invalid_argument("Unknown solver: " + solver);
        }
        
        return X;
    }
    
    // Optimized transformation function
    static MatrixXd transform(const MatrixXd& source_points,
                            const MatrixXd& control_points,
                            const MatrixXd& coefficients) {
        
        if (source_points.cols() != control_points.cols()) {
            throw std::invalid_argument("Source points and control points must have the same number of dimensions");
        }
        
        const int n = source_points.rows();
        const int d = source_points.cols();
        
        // Compute A matrix (radial basis functions)
        MatrixXd A = pairwise_radial_basis(source_points, control_points);
        
        // Construct K matrix [A, 1, source_points]
        MatrixXd K(n, A.cols() + 1 + d);
        K.leftCols(A.cols()) = A;
        K.col(A.cols()) = VectorXd::Ones(n);
        K.rightCols(d) = source_points;
        
        // Apply transformation
        return K * coefficients;
    }
};

// Pybind11 module definition
PYBIND11_MODULE(tps_cpp, m) {
    m.doc() = "Optimized TPS (Thin Plate Spline) implementation in C++";
    
    py::class_<TPSOptimized>(m, "TPSOptimized")
        .def_static("pairwise_radial_basis", &TPSOptimized::pairwise_radial_basis,
                   "Compute pairwise radial basis functions",
                   py::arg("K"), py::arg("B"))
        .def_static("find_coefficients", &TPSOptimized::find_coefficients,
                   "Find TPS coefficients",
                   py::arg("control_points"), py::arg("target_points"),
                   py::arg("lambda") = 0.0, py::arg("solver") = "exact")
        .def_static("transform", &TPSOptimized::transform,
                   "Apply TPS transformation",
                   py::arg("source_points"), py::arg("control_points"), py::arg("coefficients"));
} 