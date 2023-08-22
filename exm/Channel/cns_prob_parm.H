#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

// using namespace amrex::literals;
using namespace amrex;

struct ProbParm
{
    Real h       = 0.006845_rt ;  // channel half height (m)
    Real Re_tau  = 220.0_rt;//Re_tau = rho_w u_tau h/mu_w
    Real Re_b    = 3.0e3;     //bulk Re = rho_b ub h/mu_w
    Real Ma_b    = 1.5_rt; //originally 1.5    //mach number bulk  = ub/aw  (aw is speed of sound at wall)
    Real u_tau = 35.0_rt; // (m/s)
    Real Tw    = 500.0_rt; // wall temperature (K)
    // Real Mt0   = 0.10;           // initial turbulent mach number


    Real Rgas = 287.0_rt;// specific gas constant (J/kgK)
    // rhow, Pw and P0
    Real muw = 2.670533348e-5; // mu at 500K for air
    Real rho_w = Re_tau * muw / (u_tau*h);
    Real Pw   = rho_w*Rgas*Tw; // csal_press(rho_w,T_w,nm_csal)

    // bulk velocity and density
    Real aw = pow(1.4_rt*Rgas*Tw,0.5_rt);
    Real ub = Ma_b*aw;
    Real rho_b = Re_b*muw/(ub*h);
    Real Tb = (Pw/Rgas*rho_b);

    // body force in x
    Real fx =  rho_w*u_tau*u_tau/(h*rho_b);



};

#endif
