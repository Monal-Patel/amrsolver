#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include <CNS_index_macros.H>
#include <CNS_K.H>

using namespace amrex;
namespace PROB {

struct ProbParm
{
  Real p_l = 1.0;
  Real p_r = 0.1;
  Real rho_l = 1.0;
  Real rho_r = 0.125;
  Real u_l = 0.0;
  Real u_r = 0.0;
};

void inline inputs() {

  ParmParse pp;

  // // Simulation time
  // pp.add("max_step", max_step);
  // pp.add("stop_time", 2.0);
  // // pp.add("time_step", 0.0); # at base level, higher level time step is computed based on the number of subcycles (depending on the subcycling strategy). stop_time = max_step*time_step
  // pp.add("cfl", 0.9);

  // geometry
  // pp.add   ("geometry.coord_sys"  , 0); //0=cart, 1=RZ  2=spherical
  // pp.addarr("geometry.is_periodic", std::vector<int>{0,1,1});
  // pp.addarr("geometry.prob_lo"    , std::vector<Real>{0.0,0.0,0.0});
  // pp.addarr("geometry.prob_hi"    , std::vector<Real>{10.0,10.0,10.0});


  // // Refinement and regridding 
  // pp.addarr("amr.n_cell", std::vector<int>{64,16,16});
  // pp.add   ("amr.max_level", 0); // maximum level number allowed
  // pp.addarr("amr.ref_ratio", std::vector<int>{2,2,2}); // refinement ratio
  // pp.addarr("amr.regrid_int", std::vector<int>{2,2,2}); // how often to regrid

  // pp.addarr("amr.blocking_factor_x", std::vector<int>{8,8,8}); 
  // pp.addarr("amr.blocking_factor_y", std::vector<int>{8,8,8}); 
  // pp.addarr("amr.blocking_factor_z", std::vector<int>{8,8,8}); 

  // pp.addarr("amr.max_grid_size_x", std::vector<int>{64,64,64}); 
  // pp.addarr("amr.max_grid_size_y", std::vector<int>{64,64,64}); 
  // pp.addarr("amr.max_grid_size_z", std::vector<int>{64,64,64}); 
  
  // pp.addarr("amr.n_error_buf", std::vector<int>{2,2,2}); // n_error_buf_x
  // pp.add   ("amr.grid_eff",0.7);
  // pp.add   ("amr.v",1);
  // define ref_ratio or ref_ratio_vect (AMReX_AMRMesh.cpp l182)
  // amr.subcycling_mode = # None, Manual, Optimal (n_cycle[lev] is set dynamically), Auto (n_cycle[lev]=ref_ratio[lev])
  // fabarray.mfiter_tile_size = 8 8 8

  // Checkpoint and plotfiles
  // pp.add   ("amr.checkpoint_files_output", 0); // 0=no, 1=yes
  // pp.add   ("amr.check_file", std::string{"chk"}); // root name of checkpoint file
  // pp.add   ("amr.check_int", 100); // number of timesteps between checkpoints
  // pp.add   ("amr.plot_files_output", 1); // 0=no, 1=yes
  // pp.add   ("amr.plot_file", std::string{"./plot/plt"}); // root name of plotfile
  // pp.add   ("amr.plot_int", 2); // number of timesteps between plotfiles
  // // pp.addarr("amr.derive_plot_vars", std:vector<std::string>{"x_velocity","pressure", "x_velocity", "y_velocity", "z_velocity", "temperature"}); // list of variables to write to plotfile

  // // Misc
  // pp.add("fab.do_initval", true);
  // pp.add("fab.init_snan", true);

  // Numerical operators
  //-1 = N/A (Incase of periodic)
  // 0 = Interior           3 = Symmetry
  // 1 = Inflow             4 = SlipWall
  // 2 = Outflow            5 = NoSlipWall
  // 6 = user defined
  pp.addarr("cns.lo_bc", std::vector<int>{2,-1,-1});
  pp.addarr("cns.hi_bc", std::vector<int>{2,-1,-1});
  pp.add   ("cns.order_rk", 2); // -2, 1, 2 or 3"
  pp.add   ("cns.stages_rk", 2); // 1, 2 or 3
  pp.add   ("cns.rhs_euler", 1); // 0=false, 1=true
  pp.add   ("cns.rhs_visc", 0); // 0=false, 1=true
  pp.add   ("cns.rhs_source", 0); // 0=false, 1=true
  pp.add   ("cns.flux_euler", 0); // 0=riemann solver, 1=KEEP/AD, 2=WENO5
  pp.add   ("cns.order_keep", 4); // Order of accuracy=2, 4 or 6"
  pp.add   ("cns.art_diss", 0); // 0=none, 1=artificial dissipation
  pp.add   ("cns.nghost",2);
  pp.add   ("cns.screen_output", 1); // 0=quiet, 1=verbose
  pp.add   ("cns.verbose", 1); // 0=quiet, 1=verbose
  pp.add   ("cns.visc_type", 1); // 0=constant, 1=dynamic

  // lets add viscosity type and different flux dissipations!


  // debugging
  // pp.add("amrex.fpe_trap_invalid",1);
  // pp.add("amrex.fpe_trap_zero",1);
  // pp.add("amrex.fpe_trap_overflow",1);


}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state, amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm) {
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();

  Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
  Real Pt, rhot, uxt;
  if (x < prob_hi[0]/2) {
      Pt = prob_parm.p_l;
      rhot = prob_parm.rho_l;
      uxt = prob_parm.u_l;
  } else {
      Pt = prob_parm.p_r;
      rhot = prob_parm.rho_r;
      uxt = prob_parm.u_r;
  }
  state(i,j,k,URHO ) = rhot;
  state(i,j,k,UMX  ) = rhot*uxt;
  state(i,j,k,UMY  ) = Real(0.0);
  state(i,j,k,UMZ  ) = Real(0.0);
  Real et = Pt/(parm.eos_gamma-Real(1.0));
  state(i,j,k,UET) = et + Real(0.5)*rhot*uxt*uxt;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
void user_tagging(int i, int j, int k, auto& tagfab, const auto &sdatafab, const auto& geomdata, const ProbParm& prob_parm , int level) {

      Real dengrad_threshold = 0.5;
      amrex::Real drhox = amrex::Math::abs(sdatafab(i+1,j,k,URHO) - sdatafab(i-1,j,k,URHO))/sdatafab(i,j,k,URHO);
      if (drhox > dengrad_threshold) {
        tagfab(i,j,k) = true;
        tagfab(i+1,j,k) = true;
        tagfab(i+2,j,k) = true;
        tagfab(i+3,j,k) = true;
      }
  }


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void user_source(int i, int j, int k, const auto& state, const auto& rhs, const ProbParm& lprobparm, Parm const& parms, auto const dx) {

}

/**
 * \brief Fill external boundary conditions for ghost cells.
 *
 * @param x         ghost cell cooridinates.
 * @param dr        wall-ghost/wall-first internal distance ratio 
 * @param s_int     flow state inside of the domain.
 * @param s_ext     flow state to be filled.
 * @param idir      direction (0: x, 1: y, 2: z).
 * @param sgn       high or low boundary (1: low, -1: high).
 * @param time      time.
 * @param geomdata  domain geometry data.
 * @param prob_parm ProbParm data as defined in prob_parm.H and initialised in
 * amrex_probinit.
 * @sa CnsFillExtDir
 * @sa CnsFillExtDir::operator()
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
bcnormal(const amrex::Real x[AMREX_SPACEDIM], amrex::Real dratio, const amrex::Real s_int[NCONS],
         const amrex::Real s_refl[NCONS], amrex::Real s_ext[NCONS],
         const int idir, const int sgn, const amrex::Real time,
         amrex::GeometryData const& /*geomdata*/,  Parm const& parm, ProbParm const& prob_parm)
{
  if (idir == 1) { // ylo or yhi

    amrex::Abort("bcnormal not coded");

    // Real q_ext[NPRIM] = {0.0};
    // no-slip
    // q_ext[QU]    = -s_int[UMX]/s_int[URHO];
    // q_ext[QV]    = -s_int[UMY]/s_int[URHO];
    // q_ext[QW]    = -s_int[UMZ]/s_int[URHO];

    // // dp/dn = 0
    // amrex::Real eint_int = (s_int[UET] - 0.5*(s_int[UMX]*s_int[UMX] + s_int[UMY]*s_int[UMY] + s_int[UMZ]*s_int[UMZ])/s_int[URHO])/s_int[URHO];
    // amrex::Real p_int = (parm.eos_gamma - 1.0)*s_int[URHO]*eint_int;
    // q_ext[QPRES] = p_int;
    // // T=Twall
    // amrex::Real T_int = p_int/(parm.Rspec*s_int[URHO]); 
    // q_ext[QT]    = max(prob_parm.Tw  +  dratio*(prob_parm.Tw - T_int),50.0);
    // // rho = eos(P,T)
    // q_ext[QRHO]  = q_ext[QPRES]/(parm.Rspec*q_ext[QT]);

    // // convert prims to cons
    // s_ext[URHO] = q_ext[QRHO];
    // s_ext[UMX] = q_ext[QRHO]*q_ext[QU];
    // s_ext[UMY] = q_ext[QRHO]*q_ext[QV];
    // s_ext[UMZ] = q_ext[QRHO]*q_ext[QW];
    // amrex::Real ekin_ext = 0.5*(q_ext[QU]*q_ext[QU] + q_ext[QV]*q_ext[QV] + q_ext[QW]*q_ext[QW]); 
    // amrex::Real eint_ext = q_ext[QPRES]/(q_ext[QRHO]*(parm.eos_gamma - 1.0));
    // s_ext[UET] = q_ext[QRHO]*(eint_ext + ekin_ext);
  }
}

} // namespace prob


// // called once from amr.init
// void amrex_probinit (const int* /*init*/, const int* /*name*/, const int* /*namelen*/, const amrex_real* /*problo*/, const amrex_real* /*probhi*/)
//   {


// // PROB::h_prob_parm = new PROB::ProbParm{};
// // #if AMREX_USE_GPU
//   // PROB::d_prob_parm = (PROB::ProbParm*)The_Device_Arena()->alloc(sizeof(PROB::ProbParm));
//   // amrex::Gpu::htod_memcpy(PROB::d_prob_parm, PROB::h_prob_parm, sizeof(PROB::ProbParm));
// // #endif
// }

#endif
