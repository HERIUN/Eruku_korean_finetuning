from .functions import *
from .instance import *

# Import optimized versions if available
from .functions_optimized import *
from .instance_optimized import TPSOptimized
# Make TPSOptimized the default TPS class if optimization is available
from .functions_optimized import is_cpp_available
if is_cpp_available():
    TPS = TPSOptimized
    print("TPS module: Using optimized C++ backend by default")
else:
    print("TPS module: Using NumPy backend")
