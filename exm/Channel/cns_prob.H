#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_parm.H"
#include "cns_prob_parm.H"

using namespace amrex;

AMREX_GPU_DEVICE inline
void cns_initdata (int i, int j, int k, Array4<Real> const& state, GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm) {
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();


  // Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
  // Real Pt, rhot, uxt;


//     ! phi(nvux) = u_b*1.5*(1 - (1/(Lref)**2)*(yy-Lref)**2)
    
//     if (yy > Lref) then
//       y_temp = (2*Lref-yy-0.001)
//     else
//       y_temp = yy-0.001
//     endif
    
//     phi(nvux) = 581.54524091 + 2687.84321585*y_temp**0.5 - 593.73493364* exp(-2008.87100276*y_temp)
//     if(y_temp<0.0_wp) phi(nvux) = 500.0_wp
//     ! if(y_temp>Lref+0.001) phi(nvux) = 0.0_wp
//     phi(nvuy) = 0.0_wp    
    
// #if threedim
//     phi(nvuz) = 0.0_wp    
// #endif
    
//     phi(nvpr) = p0
//     phi(nvtp) = 677.04110513 + 255.46817*y_temp**0.5 - 176.38967315*exp(-3081.95957181*y_temp)
//     if(y_temp<0.0_wp) phi(nvtp) = T_w
//     ! if(y_temp>Lref+0.001) phi(nvtp) = T_w
//     ! phi(nvtp) = T_b
    
//     call satisfy_eos(phi)
  
  state(i,j,k,URHO ) = prob_parm.rho_b;
  state(i,j,k,UMX  ) = prob_parm.rho_b*prob_parm.ub;
  state(i,j,k,UMY  ) = Real(0.0);
  state(i,j,k,UMZ  ) = Real(0.0);
  Real rhoeint = prob_parm.Pw/(parm.eos_gamma-Real(1.0));
  state(i,j,k,UET) = rhoeint+ Real(0.5)*prob_parm.rho_b*prob_parm.ub*prob_parm.ub;

}

void tagging(TagBoxArray& tags, MultiFab &sdata, int level) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  Real dengrad_threshold= Real(0.2);
  for (MFIter mfi(tags,TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto const& tagfab = tags.array(mfi);
    auto const& sdf = sdata.array(mfi); // state data fab (sdf)
    int idx = 0; // density index
    ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept 
    {
      // Real ax = Math::abs(sdf(i+1,j,k,idx) - sdf(i,j,k,idx));
      // Real ay = Math::abs(sdf(i,j+1,k,idx) - sdf(i,j,k,idx));
      // Real az = Math::abs(sdf(i,j,k+1,idx) - sdf(i,j,k,idx));
      // ax = max(ax,Math::abs(sdf(i,j,k,idx) - sdf(i-1,j,k,idx)));
      // ay = max(ay,Math::abs(sdf(i,j,k,idx) - sdf(i,j-1,k,idx)));
      // az = max(az,Math::abs(sdf(i,j,k,idx) - sdf(i,j,k-1,idx)));
      // if (max(ax,ay,az) >= dengrad_threshold) {
      //     tagfab(i,j,k) = true;}
      Real drhox = Math::abs(sdf(i+1,j,k,idx) - sdf(i-1,j,k,idx))/sdf(i,j,k,idx);
      tagfab(i,j,k) = drhox > 0.5f;
    });
  }
}


/**
 * \brief Fill external boundary conditions for ghost cells.
 * 
 * @param x         ghost cell cooridinates.
 * @param s_int     flow state inside of the domain.
 * @param s_ext     flow state to be filled.
 * @param idir      direction (0: x, 1: y, 2: z).
 * @param sgn       high or low boundary (1: low, -1: high).
 * @param time      time.
 * @param geomdata  domain geometry data.
 * @param prob_parm ProbParm data as defined in prob_parm.H and initialised in amrex_probinit.
 * @sa CnsFillExtDir
 * @sa CnsFillExtDir::operator()
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
userbcfill (const Real x[AMREX_SPACEDIM],
          const Real s_int[NCONS],
          Real s_ext[NCONS],
          const int idir,
          const int sgn, 
          const Real time,
          GeometryData const& /*geomdata*/,
          ProbParm const& prob_parm)
{
    // for (int nf = 0; nf <= NUM_FIELD; ++nf) {
    //     if (idir == 0 && sgn == 1) { //xlo
    //         // post shock conditions
    //         s_ext[nf*NVAR + URHO] = prob_parm.rho_l;
    //         s_ext[nf*NVAR + UMX] = prob_parm.rho_l * prob_parm.u_l;
    //         s_ext[nf*NVAR + UMY] = prob_parm.rho_l * prob_parm.v_l;
    //         s_ext[nf*NVAR + UMZ] = 0.0;
    //         s_ext[nf*NVAR + UEDEN] = prob_parm.rhoe_l + 0.5 * prob_parm.rho_l * (prob_parm.u_l * prob_parm.u_l + prob_parm.v_l * prob_parm.v_l);
    //         // s_ext[nf*NVAR + UEINT] = prob_parm.rhoe_l;
    //         // s_ext[nf*NVAR + UTEMP] = prob_parm.T_l;
    //         s_ext[nf*NVAR + UFS] = prob_parm.rho_l;
    //     } else if (idir == 1) {
    //         if (sgn == 1) { //ylo
    //             if (x[0] < 1./6.) {
    //                 // FOExtrap
    //                 for (int i = 0; i < LEN_STATE; ++i) s_ext[i] = s_int[i];
    //             } else {
    //                 // SlipWall
    //                 s_ext[nf*NVAR + URHO] = s_int[URHO];
    //                 s_ext[nf*NVAR + UMX] = s_int[UMX];
    //                 s_ext[nf*NVAR + UMY] =-s_int[UMY];
    //                 s_ext[nf*NVAR + UMZ] = s_int[UMZ];
    //                 s_ext[nf*NVAR + UEDEN] = s_int[UEDEN];
    //                 // s_ext[nf*NVAR + UEINT] = s_int[UEINT];
    //                 // s_ext[nf*NVAR + UTEMP] = s_int[UTEMP];
    //                 s_ext[nf*NVAR + UFS] = s_int[UFS];
    //             }
    //         } else { //yhi
    //             // moving shock conditions
    //             Real us = 10. * prob_parm.c_r / sqrt(3.) * 2.; //shock speed
    //             if (x[0] - us*time < 1./6. + 1./sqrt(3.)*x[1]) {
    //                 s_ext[nf*NVAR + URHO] = prob_parm.rho_l;
    //                 s_ext[nf*NVAR + UMX] = prob_parm.rho_l * prob_parm.u_l;
    //                 s_ext[nf*NVAR + UMY] = prob_parm.rho_l * prob_parm.v_l;
    //                 s_ext[nf*NVAR + UMZ] = 0.0;
    //                 s_ext[nf*NVAR + UEDEN] = prob_parm.rhoe_l + 0.5 * prob_parm.rho_l * (prob_parm.u_l * prob_parm.u_l + prob_parm.v_l * prob_parm.v_l);
    //                 // s_ext[nf*NVAR + UEINT] = prob_parm.rhoe_l;
    //                 // s_ext[nf*NVAR + UTEMP] = prob_parm.T_l;
    //                 s_ext[nf*NVAR + UFS] = prob_parm.rho_l;
    //             } else {
    //                 s_ext[nf*NVAR + URHO] = prob_parm.rho_r;
    //                 s_ext[nf*NVAR + UMX] = 0.0;
    //                 s_ext[nf*NVAR + UMY] = 0.0;
    //                 s_ext[nf*NVAR + UMZ] = 0.0;
    //                 s_ext[nf*NVAR + UEDEN] = prob_parm.rhoe_r;
    //                 // s_ext[nf*NVAR + UEINT] = prob_parm.rhoe_r;
    //                 // s_ext[nf*NVAR + UTEMP] = prob_parm.T_r;
    //                 s_ext[nf*NVAR + UFS] = prob_parm.rho_r;
    //             }  
    //         }
    //     }
    // }
}
#endif
