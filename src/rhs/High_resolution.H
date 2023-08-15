#ifndef High_resolution_H
#define High_resolution_H

#include <CNS.H>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>

namespace HiRes {

  // computes Euler fluxes from conserved variable vector 
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  void cons2eulerflux(int i, int j, int k, auto const& cons, auto const& fx, auto const& fy, auto const& fz, const Parm& parm) {

    Real rhoinv = Real(1.0)/cons(i,j,k,URHO);
    Real momx   = cons(i,j,k,UMX);
    Real momy   = cons(i,j,k,UMY);
    Real momz   = cons(i,j,k,UMZ);
    Real rhoet  = cons(i,j,k,UET);
    Real ux     = momx*rhoinv;
    Real uy     = momy*rhoinv;
    Real uz     = momz*rhoinv;

    Real rhoekin = Real(0.5)*rhoinv*(momx*momx + momy*momy + momz*momz);
    Real rhoeint = rhoet - rhoekin;
    Real P       = (parm.eos_gamma - Real(1.0))*rhoeint;

    fx(i,j,k,URHO)  = momx;
    fx(i,j,k,UMX)   = momx*ux + P;
    fx(i,j,k,UMY)   = momy*ux;
    fx(i,j,k,UMZ)   = momz*ux;
    fx(i,j,k,UET)   = (rhoet + P)*ux;

    fy(i,j,k,URHO)  = momy;
    fy(i,j,k,UMX)   = momx*uy;
    fy(i,j,k,UMY)   = momy*uy + P;
    fy(i,j,k,UMZ)   = momz*uy;
    fy(i,j,k,UET)   = (rhoet + P)*uy;

    fz(i,j,k,URHO)  = momz;
    fz(i,j,k,UMX)   = momx*uz;
    fz(i,j,k,UMY)   = momy*uz;
    fz(i,j,k,UMZ)   = momz*uz + P;
    fz(i,j,k,UET)   = (rhoet + P)*uz;

    // for (int n=0;n<5;n++){
      // printf("%i %i %i %i \n",i,j,k,n);
      // printf("%i %f %f %f \n",n,fx(i,j,k,n),fy(i,j,k,n),fz(i,j,k,n));
      // }
      // printf("cons %f %f %f %f %f\n",cons(i,j,k,0), cons(i,j,k,1), cons(i,j,k,2), cons(i,j,k,3),cons(i,j,k,4));
      // printf("fx %f %f %f %f %f\n",fx(i,j,k,0), fx(i,j,k,1), fx(i,j,k,2), fx(i,j,k,3),fx(i,j,k,4));
  }

// AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
// void cons2char (int i, int j, int k, int dir, Array4<const Real> const& q, const Array4<const Real>& w, const int sys)
// {
//     int QUN, QUT, QUTT;
//     if      (dir == 0) { QUN = QU;  QUT = QV;  QUTT = QW; } 
//     else if (dir == 1) { QUN = QV;  QUT = QU;  QUTT = QW; } 
//     else               { QUN = QW;  QUT = QU;  QUTT = QV; }
    
//     if (sys == 0) { // speed of sound system
//         // Real rmpoCshock = q(i,j,k,QRHO) - q(i,j,k,QPRES)/q(i,j,k,QC)/q(i,j,k,QC); //rho minus p over c^2
//         w(i,j,k,WRHO) = q(i,j,k,QRHO) - q(i,j,k,QPRES)/q(i,j,k,QC)/q(i,j,k,QC); //rho minus p over c^2
//         w(i,j,k,WACO)   = 0.5 * (q(i,j,k,QPRES)/q(i,j,k,QC) + q(i,j,k,QRHO)*q(i,j,k,QUN));
//         w(i,j,k,WACO+1) = 0.5 * (q(i,j,k,QPRES)/q(i,j,k,QC) - q(i,j,k,QRHO)*q(i,j,k,QUN));

//     } else { // gamma system
//         w(i,j,k,WRHO) = q(i,j,k,QRHO) * (1.0 - 1.0 / q(i,j,k,QG));
//         w(i,j,k,WACO) = 0.5 * (q(i,j,k,QPRES) + std::sqrt(q(i,j,k,QG)*q(i,j,k,QRHO)*q(i,j,k,QPRES))*q(i,j,k,QUN));
//         w(i,j,k,WACO+1) = 0.5 * (q(i,j,k,QPRES) - std::sqrt(q(i,j,k,QG)*q(i,j,k,QRHO)*q(i,j,k,QPRES))*q(i,j,k,QUN));
//     }

//     // Passive scalars are the same for both systems
//     for (int n = 0; n < NUM_SPECIES; ++n) {
//         w(i,j,k,WY+n) = q(i,j,k,QFS+n);
//     }
//     w(i,j,k,WC) = q(i,j,k,QC);
//     AMREX_D_TERM(,
//         w(i,j,k,WUT)   = q(i,j,k,QUT);,
//         w(i,j,k,WUT+1) = q(i,j,k,QUTT););
// }

  // computes Euler fluxes from conserved variable vector and maxeigen value
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  void cons2eulerflux_lambda(int i, int j, int k, auto const& cons, auto const& fx, auto const& fy, auto const& fz, auto const& lambda ,const Parm& parm) {

    Real rhoinv = Real(1.0)/cons(i,j,k,URHO);
    Real momx   = cons(i,j,k,UMX);
    Real momy   = cons(i,j,k,UMY);
    Real momz   = cons(i,j,k,UMZ);
    Real rhoet  = cons(i,j,k,UET);
    Real ux     = momx*rhoinv;
    Real uy     = momy*rhoinv;
    Real uz     = momz*rhoinv;

    Real rhoekin = Real(0.5)*rhoinv*(momx*momx + momy*momy + momz*momz);
    Real rhoeint = rhoet - rhoekin;
    Real P       = (parm.eos_gamma - Real(1.0))*rhoeint;

    fx(i,j,k,URHO)  = momx;
    fx(i,j,k,UMX)   = momx*ux + P;
    fx(i,j,k,UMY)   = momy*ux;
    fx(i,j,k,UMZ)   = momz*ux;
    fx(i,j,k,UET)   = (rhoet + P)*ux;

    fy(i,j,k,URHO)  = momy;
    fy(i,j,k,UMX)   = momx*uy;
    fy(i,j,k,UMY)   = momy*uy + P;
    fy(i,j,k,UMZ)   = momz*uy;
    fy(i,j,k,UET)   = (rhoet + P)*uy;

    fz(i,j,k,URHO)  = momz;
    fz(i,j,k,UMX)   = momx*uz;
    fz(i,j,k,UMY)   = momy*uz;
    fz(i,j,k,UMZ)   = momz*uz + P;
    fz(i,j,k,UET)   = (rhoet + P)*uz;

    Real cs=sqrt(parm.eos_gamma*P*rhoinv); 
    lambda(i,j,k,0) = std::abs(max(ux+cs,ux-cs,ux)); 
    lambda(i,j,k,1) = std::abs(max(uy+cs,uy-cs,uy)); 
    lambda(i,j,k,2) = std::abs(max(uz+cs,uz-cs,uz)); 
  }

  /////////////////////////
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE
  Real weno5js(const GpuArray<Real,NCONS>& s)
  { 
    constexpr Real eps = 1e-40;
    constexpr Real q   = 2.0_rt;

    Real vl[3];
    Real beta[3];
    Real alpha[3];

    // TODO: REMOVE POW() for speed-up!
    beta[2] = (13.0_rt / 12.0_rt) * pow(s[0] - 2.0_rt * s[1] + s[2], 2) +
              0.25 * pow(s[0] - 4.0_rt * s[1] + 3.0_rt * s[2], 2);
    beta[1] = (13.0_rt / 12.0_rt) * pow(s[1] - 2.0_rt * s[2] + s[3], 2) +
              0.25 * (s[1] - s[3]) * (s[1] - s[3]);
    beta[0] = (13.0_rt / 12.0_rt) * pow(s[2] - 2.0_rt * s[3] + s[4], q) +
              0.25_rt * pow(3.0_rt * s[2] - 4.0_rt * s[3] + s[4], q);

    beta[2] = 1.0_rt / ((eps + beta[2]) * (eps + beta[2]));
    beta[1] = 1.0_rt / ((eps + beta[1]) * (eps + beta[1]));
    beta[0] = 1.0_rt / ((eps + beta[0]) * (eps + beta[0]));

    alpha[2] = beta[2];
    alpha[1] = 6.0_rt * beta[1];
    alpha[0] = 3.0_rt * beta[0];
    Real alpha1 = 1.0_rt / (alpha[2] + alpha[1] + alpha[0]);

    vl[2] = 2.0_rt * s[0] - 7.0_rt * s[1] + 11.0_rt * s[2];
    vl[1] = -s[1] + 5.0_rt * s[2] + 2.0_rt * s[3];
    vl[0] = 2.0_rt * s[2] + 5.0_rt * s[3] - s[4];

    Real sl = (Real(1.0)/6.0_rt) * alpha1 * (alpha[2] * vl[2] + alpha[1] * vl[1] + alpha[0] * vl[0]);

    return sl ;
  }

  // Computes Euler fluxes with global lax-friedrichs splitting at i-1/2,j-1/2,k-1/2
  AMREX_GPU_DEVICE inline 
  void numericalflux_globallaxsplit (int i, int j, int k, int n, const auto& cons, const auto& pfx, const auto& pfy, const auto& pfz, const auto& lambda, const auto& nfx, const auto& nfy, const auto& nfz) {

  GpuArray<Real,5> sten;
  GpuArray<Real,6> df;

  // x direction /////////////////////////////////////////////////////////
  Real maxeigen = max(lambda(i-3,j,k,0),lambda(i-2,j,k,0),lambda(i-1,j,k,0),lambda(i,j,k,0), lambda(i+1,j,k,0), lambda(i+2,j,k,0));

  df[0] = maxeigen*cons(i-3,j,k,n);
  df[1] = maxeigen*cons(i-2,j,k,n);
  df[2] = maxeigen*cons(i-1,j,k,n);
  df[3] = maxeigen*cons(i  ,j,k,n);
  df[4] = maxeigen*cons(i+1,j,k,n);
  df[5] = maxeigen*cons(i+2,j,k,n);

  // f(i-1/2)^+
  sten[0] = Real(0.5)*(pfx(i-3,j,k,n) + df[0]);
  sten[1] = Real(0.5)*(pfx(i-2,j,k,n) + df[1]);
  sten[2] = Real(0.5)*(pfx(i-1,j,k,n) + df[2]);
  sten[3] = Real(0.5)*(pfx(i  ,j,k,n) + df[3]);
  sten[4] = Real(0.5)*(pfx(i+1,j,k,n) + df[4]);
  Real flx = weno5js(sten);

  // f(i-1/2)^-
  sten[0] = Real(0.5)*(pfx(i-2,j,k,n) - df[1]);
  sten[1] = Real(0.5)*(pfx(i-1,j,k,n) - df[2]);
  sten[2] = Real(0.5)*(pfx(i  ,j,k,n) - df[3]);
  sten[3] = Real(0.5)*(pfx(i+1,j,k,n) - df[4]);
  sten[4] = Real(0.5)*(pfx(i+2,j,k,n) - df[5]);
  flx += weno5js(sten);
  nfx(i,j,k,n) = flx;

  // y direction /////////////////////////////////////////////////////////
  maxeigen = max(lambda(i,j-3,k,0),lambda(i,j-2,k,0),lambda(i,j-1,k,0),lambda(i,j,k,0), lambda(i,j+1,k,0), lambda(i,j+2,k,0));
  df[0] = maxeigen*cons(i,j-3,k,n);
  df[1] = maxeigen*cons(i,j-2,k,n);
  df[2] = maxeigen*cons(i,j-1,k,n);
  df[3] = maxeigen*cons(i,j  ,k,n);
  df[4] = maxeigen*cons(i,j+1,k,n);
  df[5] = maxeigen*cons(i,j+2,k,n);

  // f(j-1/2)^+
  sten[0] = Real(0.5)*(pfy(i,j-3,k,n) + df[0]);
  sten[1] = Real(0.5)*(pfy(i,j-2,k,n) + df[1]);
  sten[2] = Real(0.5)*(pfy(i,j-1,k,n) + df[2]);
  sten[3] = Real(0.5)*(pfy(i,j  ,k,n) + df[3]);
  sten[4] = Real(0.5)*(pfy(i,j+1,k,n) + df[4]);
  flx = weno5js(sten);

  // f(j-1/2)^-
  sten[0] = Real(0.5)*(pfy(i,j-2,k,n) - df[1]);
  sten[1] = Real(0.5)*(pfy(i,j-1,k,n) - df[2]);
  sten[2] = Real(0.5)*(pfy(i,j  ,k,n) - df[3]);
  sten[3] = Real(0.5)*(pfy(i,j+1,k,n) - df[4]);
  sten[4] = Real(0.5)*(pfy(i,j+2,k,n) - df[5]);
  flx += weno5js(sten);
  nfy(i,j,k,n) = flx;

  // z direction /////////////////////////////////////////////////////////
  maxeigen = max(lambda(i,j,k-3,0),lambda(i,j,k-2,0),lambda(i,j,k-1,0),lambda(i,j,k,0), lambda(i,j,k+1,0), lambda(i,j,k+2,0));
  df[0] = maxeigen*cons(i,j,k-3,n);
  df[1] = maxeigen*cons(i,j,k-2,n);
  df[2] = maxeigen*cons(i,j,k-1,n);
  df[3] = maxeigen*cons(i,j,k  ,n);
  df[4] = maxeigen*cons(i,j,k+1,n);
  df[5] = maxeigen*cons(i,j,k+2,n);

  // f(k-1/2)^+
  sten[0] = Real(0.5)*(pfz(i,j,k-3,n) + df[0]);
  sten[1] = Real(0.5)*(pfz(i,j,k-2,n) + df[1]);
  sten[2] = Real(0.5)*(pfz(i,j,k-1,n) + df[2]);
  sten[3] = Real(0.5)*(pfz(i,j,k  ,n) + df[3]);
  sten[4] = Real(0.5)*(pfz(i,j,k+1,n) + df[4]);
  flx = weno5js(sten);

  // f(k-1/2)^-
  sten[0] = Real(0.5)*(pfz(i,j,k-2,n) - df[1]);
  sten[1] = Real(0.5)*(pfz(i,j,k-1,n) - df[2]);
  sten[2] = Real(0.5)*(pfz(i,j,k  ,n) - df[3]);
  sten[3] = Real(0.5)*(pfz(i,j,k+1,n) - df[4]);
  sten[4] = Real(0.5)*(pfz(i,j,k+2,n) - df[5]);
  flx += weno5js(sten);
  nfz(i,j,k,n) = flx;
  }

  // Computes Euler fluxes at i-1/2,j-1/2,k-1/2
  AMREX_GPU_DEVICE inline 
  void numericalflux (int i, int j, int k, int n, const auto& pfx, const auto& pfy, const auto& pfz, const auto& nfx, const auto& nfy, const auto& nfz) {

  GpuArray<Real,5> sten;

  // i-1/2
  sten[0] = pfx(i-3,j,k,n);
  sten[1] = pfx(i-2,j,k,n);
  sten[2] = pfx(i-1,j,k,n);
  sten[3] = pfx(i  ,j,k,n);
  sten[4] = pfx(i+1,j,k,n);
  nfx(i,j,k,n) = weno5js(sten);

  // j-1/2
  sten[0] = pfy(i,j-3,k,n);
  sten[1] = pfy(i,j-2,k,n);
  sten[2] = pfy(i,j-1,k,n);
  sten[3] = pfy(i,j  ,k,n);
  sten[4] = pfy(i,j+1,k,n);
  nfy(i,j,k,n) = weno5js(sten);

  // k-1/2
  sten[0] = pfz(i,j,k-3,n);
  sten[1] = pfz(i,j,k-2,n);
  sten[2] = pfz(i,j,k-1,n);
  sten[3] = pfz(i,j,k  ,n);
  sten[4] = pfz(i,j,k+1,n);
  nfz(i,j,k,n) = weno5js(sten);
  }

  inline void FluxWENO(MultiFab& statemf,  MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {
    // make multifab for variables
    Array<MultiFab,AMREX_SPACEDIM> pntflxmf;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      pntflxmf[idim].define( statemf.boxArray(), statemf.DistributionMap(), NCONS, CNS::NGHOST);}

    //store eigenvalues max(u+c,u,u-c) in all directions
    MultiFab lambdamf;
    lambdamf.define( statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, CNS::NGHOST);

    Parm& lparm = *CNS::d_parm;
    // loop over all fabs
    for (MFIter mfi(statemf, false); mfi.isValid(); ++mfi)
    {
        const Box& bxg     = mfi.growntilebox(CNS::NGHOST);
        const Box& bxnodal = mfi.grownnodaltilebox(-1,0); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

        auto const& statefab = statemf.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                     auto const& nfabfy = numflxmf[1].array(mfi);,
                     auto const& nfabfz = numflxmf[2].array(mfi););

        AMREX_D_TERM(auto const& pfabfx = pntflxmf[0].array(mfi);,
                     auto const& pfabfy = pntflxmf[1].array(mfi);,
                     auto const& pfabfz = pntflxmf[2].array(mfi););

        auto const& lambda = lambdamf.array(mfi);

        ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            cons2eulerflux_lambda(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lambda ,lparm);
        });

        // ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        //     cons2char(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lparm);
        // });

        ParallelFor(bxnodal, int(NCONS) , [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
              numericalflux_globallaxsplit(i, j, k, n, statefab ,pfabfx, pfabfy, pfabfz, lambda ,nfabfx, nfabfy, nfabfz); 
        });
    }
  }
}
#endif