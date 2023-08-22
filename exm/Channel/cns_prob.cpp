
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "CNS_index_macros.H"
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
  void amrex_probinit (const int* /*init*/, const int* /*name*/, const int* /*namelen*/, const amrex_real* /*problo*/, const amrex_real* /*probhi*/)
  {

    CNS::h_parm->Cshock = 0.0_rt;
    CNS::h_parm->Cdamp  = 0.0001_rt;

    CNS::d_parm->Cshock = CNS::h_parm->Cshock;
    CNS::d_parm->Cdamp  = CNS::h_parm->Cdamp;


    Print() << CNS::h_prob_parm->muw << std::endl;
    // Print() << CNS::h_parm->visc_CPU(500) << std::endl;
    // Print() << CNS::h_parm->cond_CPU(500) << std::endl;
    Print() << CNS::h_parm->cp << std::endl;
    // Print() << CNS::h_parm->visc_CPU(500)*CNS::h_parm->cp/ CNS::h_parm->cond_CPU(500)<< std::endl;
    Print() << CNS::h_prob_parm->rho_w << std::endl;
    Print() << CNS::h_prob_parm->Pw << std::endl;
    exit(0);
    // print lv_w , dy^+, dx^+, dz^+

  }

}
