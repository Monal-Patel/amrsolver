TINY_PROFILE = FALSE
DEBUG = TRUE
DIM=3
USE_MPI = TRUE
USE_OMP = FALSE
USE_GPIBM=TRUE

include ./Make.package
include $(AMR_SOLVER)/src/Make.CNS

INCLUDE_LOCATIONS += $(AMR_SOLVER)/lib/install/boost/include
LIBRARY_LOCATIONS += $(AMR_SOLVER)/lib/install/boost/lib

INCLUDE_LOCATIONS += $(AMR_SOLVER)/lib/install/cgal/include
LIBRARY_LOCATIONS += $(AMR_SOLVER)/lib/install/cgal/lib/cmake/CGAL
LIBRARIES += -lgmp