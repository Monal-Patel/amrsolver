TINY_PROFILE = FALSE
DEBUG = FALSE
DIM=3
USE_MPI = TRUE
USE_CUDA= FALSE
USE_OMP = FALSE
USE_GPIBM=TRUE
AMREX_NO_PROBINIT=TRUE

include ./Make.package
include $(AMR_SOLVER)/src/Make.CNS

INCLUDE_LOCATIONS += $(AMREX_HOME)/../install/boost/include
LIBRARY_LOCATIONS += $(AMREX_HOME)/../install/boost/lib

INCLUDE_LOCATIONS += $(AMREX_HOME)/../install/cgal/include
LIBRARY_LOCATIONS += $(AMREX_HOME)/../install/cgal/lib/cmake/CGAL
LIBRARIES += -lgmp
