
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "CNS_index_macros.H"
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
  void amrex_probinit (const int* /*init*/, const int* /*name*/, const int* /*namelen*/, const amrex_real* /*problo*/, const amrex_real* /*probhi*/)
  {

    CNS::h_parm->Cshock = 0.0_rt;
    CNS::h_parm->Cdamp  = 0.0016_rt;

    CNS::d_parm->Cshock = CNS::h_parm->Cshock;
    CNS::d_parm->Cdamp  = CNS::h_parm->Cdamp;

  }

}
