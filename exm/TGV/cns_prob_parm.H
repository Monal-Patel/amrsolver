#ifndef CNS_PROB_PARM_H
#define CNS_PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct ProbParm
{
  bool convecting = false;

  amrex::Real reynolds = amrex::Real(1600.0);
  amrex::Real mach = amrex::Real(0.1);
  amrex::Real prandtl = amrex::Real(0.71);
  amrex::Real omega_x = amrex::Real(1.0); // [rad s^-1]
  amrex::Real omega_y = amrex::Real(1.0); // [rad s^-1]
  amrex::Real omega_z = amrex::Real(1.0); // [rad s^-1]
  amrex::Real L  = amrex::Real(1.0);       // [m]
  amrex::Real T0 = amrex::Real(110.4);    // [K]
  amrex::Real p0 = 0.0;  // [Pa]
  amrex::Real rho0 = 0.0;    // [Kg m^-3]
  amrex::Real v0 = 0.0;  // [m s^-1]

  amrex:: Real Tw = 0.0; // not used but needed because of how BCs are implemented currently --> TO IMPROVE
};

#endif
