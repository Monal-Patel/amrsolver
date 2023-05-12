#ifndef CNS_PARM_H_
#define CNS_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct Parm
{
    amrex::Real eos_gamma = 1.4;
    amrex::Real eos_mu = 28.97e-3;// mean molecular weight kg/mol

    //amrex::Real Pr  = 0.72;     // Prandtl number
    //Sutherland's fit from Computational Fluid Mechanics and Heat Transfer 
    amrex::Real visc_ref  = 1.458e-5; // Viscosity reference value
    amrex::Real Tvisc_ref = 110.4;    // Viscosity reference temperature
    amrex::Real cond_ref  = 2.495e-3; // Conductivity reference value
    amrex::Real Tcond_ref = 194.0;    // Conductivity reference temperature

    amrex::Real smallr = 1.e-19;
    amrex::Real smallp = 1.e-10;

    amrex::Real Ru = amrex::Real(8.31451);
    amrex::Real cv = Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    amrex::Real cp = eos_gamma * Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    amrex::Real Rspec = Ru/eos_mu;

    public: 
    // AMREX_GPU_DEVICE inline const amrex::Real visc(amrex::Real& T,mu) { 
    AMREX_GPU_DEVICE inline amrex::Real visc(amrex::Real& T) const { return visc_ref*pow(T,amrex::Real(1.5))/(Tvisc_ref + T);}

    AMREX_GPU_DEVICE inline amrex::Real cond(amrex::Real& T) const { return cond_ref*pow(T,amrex::Real(1.5))/(Tcond_ref + T);}
};

// struct ProbParm
// {
//     amrex::Real p_l = 1.0;
//     amrex::Real p_r = 0.1;
//     amrex::Real rho_l = 1.0;
//     amrex::Real rho_r = 0.125;
//     amrex::Real u_l = 0.0;
//     amrex::Real u_r = 0.0;
// };
#endif
