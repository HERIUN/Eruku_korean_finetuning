import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import numpy as np
import os
import sys

# Find Eigen include directories
def find_eigen_dirs():
    possible_dirs = [
        os.path.join(os.path.dirname(__file__), 'eigen-3.4.0'),  # Local Eigen installation
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include', 'eigen3'),  # Conda environment
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include'),  # Conda environment without eigen3
        "/usr/include/eigen3",  # Common location for Eigen on Linux
        "/usr/local/include/eigen3",  # Alternative location
        "/opt/homebrew/include/eigen3",  # macOS with Homebrew
    ]
    
    # Filter to existing directories
    existing_dirs = [d for d in possible_dirs if os.path.exists(d)]
    if not existing_dirs:
        raise RuntimeError("Eigen not found. Please install Eigen and ensure it's in your include path.")
    return existing_dirs

eigen_dirs = find_eigen_dirs()
print(f"Found Eigen directories: {eigen_dirs}")

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "tps_cpp",
        sources=["tps_cpp.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
            np.get_include(),
        ] + eigen_dirs,
        language="c++",
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=[],
    ),
]

setup(
    name="tps_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
) 