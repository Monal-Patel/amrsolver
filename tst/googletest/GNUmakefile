TINY_PROFILE = FALSE
DEBUG = TRUE
DIM=3
USE_MPI = TRUE
USE_OMP = FALSE
USE_GPIBM=FALSE

include ./Make.package
include ./Make.CNS

INCLUDE_LOCATIONS += $(AMR_SOLVER)/lib/install/boost/include
LIBRARY_LOCATIONS += $(AMR_SOLVER)/lib/install/boost/lib

INCLUDE_LOCATIONS += $(AMR_SOLVER)/lib/install/cgal/include
LIBRARY_LOCATIONS += $(AMR_SOLVER)/lib/install/cgal/lib/cmake/CGAL

INCLUDE_LOCATIONS += $(AMR_SOLVER)/lib/install/googletest/include
LIBRARY_LOCATIONS += $(AMR_SOLVER)/lib/install/googletest/lib

LIBRARIES += -lgmp -lgtest