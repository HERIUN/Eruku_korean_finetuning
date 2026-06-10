#!/bin/bash

# Build script for TPS C++ optimization module

set -e  # Exit on any error

echo "=== Building Optimized TPS Module ==="

# Check if we're in the right directory
if [ ! -f "tps_cpp.cpp" ]; then
    echo "Error: tps_cpp.cpp not found. Please run this script from the tps directory."
    exit 1
fi

echo "1. Installing/upgrading required packages..."

# Check for system dependencies (Eigen)
if ! pkg-config --exists eigen3; then
    echo "⚠️  Eigen3 not found via pkg-config"
    echo "Please install Eigen3 manually:"
    echo "  - Ubuntu/Debian: sudo apt-get install libeigen3-dev"
    echo "  - Or download from: https://eigen.tuxfamily.org/"
    echo "  - Or use conda: conda install eigen"
    echo ""
    echo "Continuing build anyway - may work if Eigen3 is in standard locations..."
else
    echo "✓ Eigen3 found"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "2. Building C++ extension..."

# Clean any previous builds
rm -rf build/ dist/ *.egg-info/ *.so

# Build the extension
python setup.py build_ext --inplace

echo "3. Testing the build..."

# Test if the module can be imported
python -c "
try:
    import tps_cpp
    print('✓ C++ module built successfully and can be imported')
except ImportError as e:
    print('✗ Failed to import C++ module:', e)
    exit(1)
"

echo "=== Build completed successfully! ==="
echo ""
echo "To use the optimized TPS module in your code:"
echo "  from custom_datasets.font_square.tps.instance_optimized import TPSOptimized"
echo "  # Use TPSOptimized instead of TPS for better performance"
echo ""
echo "To test performance improvements:"
echo "  python test_performance.py" 