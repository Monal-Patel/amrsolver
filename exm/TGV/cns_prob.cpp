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
    amrex::ParmParse pp("prob");
    pp.query("reynolds", CNS::h_prob_parm->reynolds);
    pp.query("mach", CNS::h_prob_parm->mach);
    pp.query("prandtl", CNS::h_prob_parm->prandtl);
    pp.query("convecting", CNS::h_prob_parm->convecting);
    pp.query("omega_x", CNS::h_prob_parm->omega_x);
    pp.query("omega_y", CNS::h_prob_parm->omega_y);
    pp.query("omega_z", CNS::h_prob_parm->omega_z);
  }

  // Initial state 
  // speed of sound and velocity
  Real cs = std::sqrt(CNS::h_parm->eos_gamma * CNS::h_parm->Rspec * CNS::h_prob_parm->T0);
  CNS::h_prob_parm->v0   = CNS::h_prob_parm->mach * cs;

  // density (rho0 = Re*mu0/(L*U0)
  CNS::h_prob_parm->rho0 = CNS::h_prob_parm->reynolds* CNS::h_parm->visc_ref/(CNS::h_prob_parm->L* CNS::h_prob_parm->v0);
  // CNS::h_prob_parm->rho0 = CNS::h_prob_parm->p0/ (CNS::h_parm->Rspec * CNS::h_prob_parm->T0);

  // pressure (p0 = rho0*R*T0)
  CNS::h_prob_parm->p0   = CNS::h_prob_parm->rho0 * CNS::h_parm->Rspec * CNS::h_prob_parm->T0;

  // amrex::Print() << "V0 = " << CNS::h_prob_parm->v0 << " p0 = " << CNS::h_prob_parm->p0 << " rho0 = " << CNS::h_prob_parm->rho0 << " T0 = " << CNS::h_prob_parm->T0 << std::endl;
}



}