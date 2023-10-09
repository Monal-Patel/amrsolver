TINY_PROFILE = FALSE
DEBUG    = TRUE
DIM      = 3
USE_MPI  = TRUE
USE_OMP  = FALSE
USE_GPIBM= TRUE
GTEST    = TRUE

include ./Make.package
include $(AMR_SOLVER)/src/Make.CNS

INCLUDE_LOCATIONS += $(AMREX_HOME)/../install/boost/include
LIBRARY_LOCATIONS += $(AMREX_HOME)/../install/boost/lib

INCLUDE_LOCATIONS += $(AMREX_HOME)/../install/cgal/include
LIBRARY_LOCATIONS += $(AMREX_HOME)/../install/cgal/lib/cmake/CGAL

INCLUDE_LOCATIONS += $(AMREX_HOME)/../install/googletest/include
LIBRARY_LOCATIONS += $(AMREX_HOME)/../install/googletest/lib

LIBRARIES += -lgmp -lgtest