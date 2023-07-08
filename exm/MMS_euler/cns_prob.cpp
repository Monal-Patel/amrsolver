#include "cns_prob.H"

extern "C" {
void
amrex_probinit(
  const int* /*init*/,
  const int* /*name*/,
  const int* /*namelen*/,
  const amrex::Real* problo,
  const amrex::Real* probhi)
{
  // Parse params
  {
    // amrex::ParmParse pp("prob");
    // pp.query("reynolds", CNS::d_prob_parm->reynolds);
    // pp.query("mach", CNS::d_prob_parm->mach);
    // pp.query("prandtl", CNS::d_prob_parm->prandtl);
    // pp.query("convecting", CNS::d_prob_parm->convecting);
    // pp.query("omega_x", CNS::d_prob_parm->omega_x);
    // pp.query("omega_y", CNS::d_prob_parm->omega_y);
    // pp.query("omega_z", CNS::d_prob_parm->omega_z);
  }

  // Initial state 

}



}