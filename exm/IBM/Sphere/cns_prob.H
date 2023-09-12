#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <CNS_index_macros.H>
#include <AMReX_ParmParse.H>
#include <AMReX_AmrLevel.H>
#include <Closures.H>

namespace PROB {

//////////////////////////// Physical closures /////////////////////////////////
struct ProbParm
{ 
  Real p_l = 1.0;
  Real p_r = 0.1;
  Real rho_l = 1.0;
  Real rho_r = 0.125;
  Real u_l = 0.0;
  Real u_r = 0.0;
};

typedef closures_derived_base_t<visc_suth_t, cond_suth_t, calorifically_perfect_gas_t, ProbParm> ProbClosures;

//////////////////////////// Numerical operators ///////////////////////////////
void inline inputs() {
  ParmParse pp;

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

  // ibm
  pp.add   ("ib.move",0); // 0=false, 1=true
  pp.add   ("ib.plot_surf",0); // 0=false, 1=true

}

//////////////////////////// Initial conditions ////////////////////////////////
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state, amrex::GeometryData const& geomdata, ProbClosures const& closures, ProbParm const& prob_parm) {
  using amrex::Real;
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();

  Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
  Real Pt, rhot, uxt;
  if (x < prob_hi[0]/10) {
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
  Real et = Pt/(closures.gamma-Real(1.0));
  state(i,j,k,UET) = et + Real(0.5)*rhot*uxt*uxt;
}

//
AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
void user_tagging(int i, int j, int k, auto& tagfab, const auto &sdatafab, const auto &ibfab, const auto& geomdata, const ProbParm& prob_parm , int level) {

      Real dengrad_threshold = 0.5;
      amrex::Real drhox = amrex::Math::abs(sdatafab(i+1,j,k,URHO) - sdatafab(i-1,j,k,URHO))/sdatafab(i,j,k,URHO);
      if (drhox > dengrad_threshold) {
        tagfab(i,j,k) = true;
        tagfab(i+1,j,k) = true;
        tagfab(i+2,j,k) = true;
        tagfab(i+3,j,k) = true;
      }
  }

// void user_tagging(amrex::TagBoxArray& tags, amrex::MultiFab& sdata, int level, IBM::IBMultiFab* ibdata){

// #ifdef AMREX_USE_OMP
// #pragma omp parallel if (Gpu::notInLaunchRegion())
// #endif
//     for (amrex::MFIter mfi(tags,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//     {
//         const amrex::Box& bx = mfi.tilebox();
//         auto const& tagfab = tags.array(mfi);
//         auto const& sdatafab = sdata.array(mfi);

//         if (ibdata!=nullptr) {
//           auto const& ibdatafab = ibdata->array(mfi);
//           amrex::ParallelFor(bx,
//           [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//           {
//             // tagging on ghs
//             tagfab(i,j,k) = ibdatafab(i,j,k,1);
//             // or fluid neighbour
//             bool condition = false; 
//             for (int l=-1; l<=1 ; l+=2) {
//               condition = condition || ibdatafab(i+l,j,k,0);
//               condition = condition || ibdatafab(i,j+l,k,0);
//               condition = condition || ibdatafab(i,j,k+l,0);
//             }
//             condition = condition && !ibdatafab(i,j,k,0);
//             tagfab(i,j,k) = tagfab(i,j,k) || condition   ;
//           });}

//         else  {
//           // Temporary tagging on density in x only
//           amrex::ParallelFor(bx,
//           [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//           {
//           // amrex::Real drhox = amrex::Math::abs(sdatafab(i+1,j,k,0) - sdatafab(i-1,j,k,0))/sdatafab(i,j,k,0);
//           // tagfab(i,j,k) = drhox > 0.5f;
//           });
//         };
//         };


//////////////////////////// Boundary conditions ///////////////////////////////
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
         amrex::GeometryData const& /*geomdata*/,  ProbClosures const& closures, ProbParm const& prob_parm)
{
  // if (idir == 1) { // ylo or yhi

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
  // }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void user_source(int i, int j, int k, const auto& state, const auto& rhs, const ProbParm& lprobparm, ProbClosures const& closures, auto const dx) {
};

}
#endif