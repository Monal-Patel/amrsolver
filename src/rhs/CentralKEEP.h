#ifndef CentralKEEP_H_
#define CentralKEEP_H_

#include <CNS.h>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>
#include <prob.h>
#ifdef AMREX_USE_GPIBM
#include <IBM.h>
#endif

using namespace amrex;

namespace CentralKEEP {

  // Could have a CentralKEEP class and declare an extern instance of it here.Then define it in CNS.cpp? 
  // Maybe better to have a 'flux' instance of type 'scheme' in CNS.cpp?

  inline int order_keep;
  //2 * standard finite difference coefficients
  inline Gpu::ManagedVector<Array1D<Real,0,2>> Vcoeffs;

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  Real fDiv(Real f,Real fl) {
    return 0.5_rt*(f + fl);}

  // divergence split
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  Real fgDiv(Real f,Real fl, Real g, Real gl) {
    return 0.5_rt*(f*gl + fl*g);}

  //Quadratic split
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  Real fgQuad(Real f,Real fl, Real g, Real gl) {
    return 0.25_rt*(f + fl)*(g + gl);}

  //Cubic split
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  Real fghCubic(Real f,Real fl, Real g, Real gl,Real h, Real hl) {
    return 0.1250_rt*(f + fl)*(g + gl)*(h + hl);}

  // Computes fluxes at i-1/2, j-1/2 and k-1/2
  // Computational cost can be reduced by computing and storing flux averages between l and m points
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  void KEEP(int i, int j, int k ,int halfsten, const auto& Vcoeffs, const auto& prims,const auto& nfabfx,const auto& nfabfy, const auto& nfabfz, PROB::ProbClosures const& closures) {

    int i1, i2, j1, j2, k1, k2;
    Real rho1, ux1, uy1, uz1, ie1, p1;
    Real rho2, ux2, uy2, uz2, ie2, p2;
    Real massflx,ke;

    Array1D<Real,0,2>& coeffs = Vcoeffs[halfsten-1]; // get coefficients array for current scheme

    // For computational efficiency here, prims(i,j,k,:) could be simplified to prims(:) to avoid accessing the large array repeatedly. Same principle could be applied to nfabfx, nfabfy and nfabfz.

    for (int n=0;n<NCONS;n++) {
      nfabfx(i,j,k,n) = 0.0_rt;
      nfabfy(i,j,k,n) = 0.0_rt;
      nfabfz(i,j,k,n) = 0.0_rt;
    }

    // TODO: add an abstraction layer, stencils, to avoid code duplication
    // flux at i-1/2
    for (int l=1; l<=halfsten; l++) {
      for (int m=0; m<=l-1; m++) {
        i1 = i + m;
        rho1 = prims(i1,j,k,QRHO);
        ux1  = prims(i1,j,k,QU);
        uy1  = prims(i1,j,k,QV);
        uz1  = prims(i1,j,k,QW);
        ie1  = closures.cv*prims(i1,j,k,QT);
        p1   = prims(i1,j,k,QPRES);

        i2   = i + m - l;
        rho2 = prims(i2,j,k,QRHO);
        ux2  = prims(i2,j,k,QU);
        uy2  = prims(i2,j,k,QV);
        uz2  = prims(i2,j,k,QW);
        ie2  = closures.cv*prims(i2,j,k,QT);
        p2   = prims(i2,j,k,QPRES);

        massflx = fgQuad(rho1,rho2,ux1,ux2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);

        nfabfx(i,j,k,URHO) += coeffs(l-1)*massflx;
        nfabfx(i,j,k,UMX ) += coeffs(l-1)*(fghCubic(rho1,rho2,ux1,ux2,ux1,ux2) +fDiv(p1,p2));
        nfabfx(i,j,k,UMY ) += coeffs(l-1)*fghCubic(rho1,rho2,ux1,ux2,uy1,uy2);
        nfabfx(i,j,k,UMZ ) += coeffs(l-1)*fghCubic(rho1,rho2,ux1,ux2,uz1,uz2);
        nfabfx(i,j,k,UET ) += coeffs(l-1)*(massflx*ke + fghCubic(rho1,rho2,ux1,ux2,ie1,ie2) + fgDiv(p1,p2,ux1,ux2));
      }
    }
    
    // flux at j-1/2
    for (int l=1; l<=halfsten; l++) {
      for (int m=0; m<=l-1; m++) {
        j1 = j + m;
        rho1 = prims(i,j1,k,QRHO);
        ux1  = prims(i,j1,k,QU);
        uy1  = prims(i,j1,k,QV);
        uz1  = prims(i,j1,k,QW);
        ie1  = closures.cv*prims(i,j1,k,QT);
        p1   = prims(i,j1,k,QPRES);

        j2   = j + m - l;
        rho2 = prims(i,j2,k,QRHO);
        ux2  = prims(i,j2,k,QU);
        uy2  = prims(i,j2,k,QV);
        uz2  = prims(i,j2,k,QW);
        ie2  = closures.cv*prims(i,j2,k,QT);
        p2   = prims(i,j2,k,QPRES);

        massflx = fgQuad(rho1,rho2,uy1,uy2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);
        nfabfy(i,j,k,URHO) += coeffs(l-1)*massflx;
        nfabfy(i,j,k,UMX ) += coeffs(l-1)*fghCubic(rho1,rho2,uy1,uy2,ux1,ux2);
        nfabfy(i,j,k,UMY ) += coeffs(l-1)*(fghCubic(rho1,rho2,uy1,uy2,uy1,uy2)+ fDiv(p1,p2));
        nfabfy(i,j,k,UMZ ) += coeffs(l-1)*fghCubic(rho1,rho2,uy1,uy2,uz1,uz2);
        nfabfy(i,j,k,UET ) += coeffs(l-1)*(massflx*ke + fghCubic(rho1,rho2,uy1,uy2,ie1,ie2)+ fgDiv(p1,p2,uy1,uy2));
      }
    }

    // flux at k-1/2
    for (int l=1; l<=halfsten; l++) {
      for (int m=0; m<=l-1; m++) {
        k1 = k + m;
        rho1 = prims(i,j,k1,QRHO);
        ux1  = prims(i,j,k1,QU);
        uy1  = prims(i,j,k1,QV);
        uz1  = prims(i,j,k1,QW);
        ie1  = closures.cv*prims(i,j,k1,QT);
        p1   = prims(i,j,k1,QPRES);

        k2   = k + m - l;
        rho2 = prims(i,j,k2,QRHO);
        ux2  = prims(i,j,k2,QU);
        uy2  = prims(i,j,k2,QV);
        uz2  = prims(i,j,k2,QW);
        ie2  = closures.cv*prims(i,j,k2,QT);
        p2   = prims(i,j,k2,QPRES);

        massflx = fgQuad(rho1,rho2,uz1,uz2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);
        nfabfz(i,j,k,URHO) += coeffs(l-1)*massflx;
        nfabfz(i,j,k,UMX)  += coeffs(l-1)*fghCubic(rho1,rho2,uz1,uz2,ux1,ux2);
        nfabfz(i,j,k,UMY)  += coeffs(l-1)*fghCubic(rho1,rho2,uz1,uz2,uy1,uy2);
        nfabfz(i,j,k,UMZ)  += coeffs(l-1)*(fghCubic(rho1,rho2,uz1,uz2,uz1,uz2) + fDiv(p1,p2));
        nfabfz(i,j,k,UET)  += coeffs(l-1)*(massflx*ke + fghCubic(rho1,rho2,uz1,uz2,ie1,ie2)+ fgDiv(p1,p2,uz1,uz2));
      }
    }

  }

  ///////////
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  void KEEPdir(int i, int j, int k, int dir ,int halfsten, const auto& Vcoeffs, const auto& prims, const auto& nflux, PROB::ProbClosures const& parm) {

    GpuArray<Real,NCONS> flx; 
    for (int n=0; n<NCONS; n++) {flx[n]=0.0;};
    GpuArray<int,3> vdir={int(dir==0),int(dir==1),int(dir==2)};

    int i1, j1, i2, j2, k1, k2;
    Real rho1, ux1, uy1, uz1, uu1, ie1, p1;
    Real rho2, ux2, uy2, uz2, uu2, ie2, p2;
    Real massflx,ke;
    Array1D<Real,0,2>& coeffs = Vcoeffs[halfsten-1]; // get coefficients array for current scheme

    // flux at i/j/k-1/2
    for (int l=1; l<=halfsten; l++) {
      for (int m=0; m<=l-1; m++) {
        i1 = i + m*vdir[0];
        j1 = j + m*vdir[1];
        k1 = k + m*vdir[2];
        rho1 = prims(i1,j1,k1,QRHO);
        ux1  = prims(i1,j1,k1,QU);
        uy1  = prims(i1,j1,k1,QV);
        uz1  = prims(i1,j1,k1,QW);
        uu1  = prims(i1,j1,k1,QU + dir);
        ie1  = parm.cv*prims(i1,j1,k1,QT);
        p1   = prims(i1,j1,k1,QPRES);

        i2   = i + (m - l)*vdir[0];
        j2   = j + (m - l)*vdir[1];
        k2   = k + (m - l)*vdir[2];
        rho2 = prims(i2,j2,k2,QRHO);
        ux2  = prims(i2,j2,k2,QU);
        uy2  = prims(i2,j2,k2,QV);
        uz2  = prims(i2,j2,k2,QW);
        ie2  = parm.cv*prims(i2,j2,k2,QT);
        p2   = prims(i2,j2,k2,QPRES);
        uu2  = prims(i2,j2,k2,QU + dir);

        massflx = fgQuad(rho1,rho2,uu1,uu2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);
        flx[URHO] = flx[URHO] + coeffs(l-1)*massflx;
        flx[UMX]  = flx[UMX]  + coeffs(l-1)*fghCubic(rho1,rho2,uu1,uu2,ux1,ux2);
        flx[UMY]  = flx[UMY]  + coeffs(l-1)*(fghCubic(rho1,rho2,uu1,uu2,uy1,uy2) + fDiv(p1,p2));
        flx[UMZ]  = flx[UMZ]  + coeffs(l-1)*fghCubic(rho1,rho2,uu1,uu2,uz1,uz2);
        flx[UET]  = flx[UET]  + coeffs(l-1)*(massflx*ke + fghCubic(rho1,rho2,uu1,uu2,ie1,ie2)+ fgDiv(p1,p2,uu1,uu2));
      }
    }
    for (int n=0; n<NCONS; n++) {nflux(i,j,k,n) = flx[n];}
  
  }


  AMREX_FORCE_INLINE void eflux(MultiFab& statemf,  MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {

    PROB::ProbClosures& lclosures = *CNS::d_prob_closures;
    auto const pVcoeffs = Vcoeffs.data();
    int order = order_keep; // redefined here so can be captured by lambda

    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bxnodal = mfi.grownnodaltilebox(-1,0);

      // auto const& statefab = statemf.array(mfi);
      auto const& prims    = primsmf.array(mfi);
      AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                    auto const& nfabfy = numflxmf[1].array(mfi);,
                    auto const& nfabfz = numflxmf[2].array(mfi););

      ParallelFor(bxnodal,  
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        KEEP(i,j,k,order/2,pVcoeffs,prims,nfabfx,nfabfy,nfabfz,lclosures);
      });
    }
  }

  // TODO: In post_init select the boundaries and planes to reduce order/treat specially -- function to generate box arrays of planes with associated order of numericalscheme to be applied.
  AMREX_FORCE_INLINE void Flux_2nd_Order_KEEP(Geometry& geom, MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {

    BCRec& l_phys_bc = *CNS::d_phys_bc;
    PROB::ProbClosures& lclosures = *CNS::d_prob_closures;
    auto const pVcoeffs = Vcoeffs.data();
    /////////////// 

    for (MFIter mfi(primsmf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bx      = mfi.tilebox();
      const Box& bxnodal = mfi.grownnodaltilebox(-1,0);
      auto const& prims  = primsmf.array(mfi);
      auto const& nfabfy = numflxmf[1].array(mfi);

      // Order reduction at ymin wall boundary
      if (l_phys_bc.lo(1)==6) {
        if (geom.Domain().smallEnd(1)==bx.smallEnd(1)) {
          int jj = bx.smallEnd(1);
          IntVect small = {bxnodal.smallEnd(0), jj, bxnodal.smallEnd(2)};
          IntVect big   = {bxnodal.bigEnd(0)  , jj + (order_keep/2) -1, bxnodal.bigEnd(2)  };
          Box bxboundary(small,big);

          ParallelFor(bxboundary, 
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            KEEPdir(i,j,k,1,1,pVcoeffs,prims,nfabfy,lclosures);
          });
        }
      }
    
    // Order reduction at ymax wall boundary
    if (l_phys_bc.hi(1)==6) {
      if (geom.Domain().bigEnd(1)==bx.bigEnd(1)) {
          int jj = bx.bigEnd(1);
          IntVect small = {bxnodal.smallEnd(0), jj - (order_keep/2) + 1, bxnodal.smallEnd(2)};
          IntVect big   = {bxnodal.bigEnd(0)  , jj, bxnodal.bigEnd(2)  };
          Box bxboundary(small,big);

          ParallelFor(bxboundary, 
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            KEEPdir(i,j,k,1,1,pVcoeffs,prims,nfabfy,lclosures);
          });
        }
      }
    }
  }


#if AMREX_USE_GPIBM
AMREX_FORCE_INLINE void ibm_flux_correction(MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf, IBM::IBMultiFab& ibmf) {

  PROB::ProbClosures& lclosures = *CNS::d_prob_closures;
  auto const pVcoeffs = Vcoeffs.data();
  int order = order_keep; // redefined here so can be captured by lambda

  for (MFIter mfi(primsmf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    auto const& ibFab   = ibmf.get(mfi);
    auto const& markers = ibmf.array(mfi);
    auto const gp_ijk   = ibFab.gpData.gp_ijk.data();
    auto const& prims    = primsmf.array(mfi);
    AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                  auto const& nfabfy = numflxmf[1].array(mfi);,
                  auto const& nfabfz = numflxmf[2].array(mfi););

    int halfsten = 1; // 2nd order
    ParallelFor(ibFab.gpData.ngps, [=] AMREX_GPU_DEVICE (int ii) noexcept
    {
      int i = gp_ijk[ii](0);
      int j = gp_ijk[ii](1);
      int k = gp_ijk[ii](2);

      if (!markers(i+1,j,k,0)) { //the point to the right is fluid then compute fi-1/2 at i+1
        KEEPdir(i+1,j,k,0,halfsten,pVcoeffs,prims,nfabfx,lclosures);
      }

      if (!markers(i-1,j,k,0)) { //the point to the left is fluid then compute fi-1/2 at i
        KEEPdir(i,j,k,0,halfsten,pVcoeffs,prims,nfabfx,lclosures);
      }

      if (!markers(i,j+1,k,0)) { //the point to the top is fluid then compute fj-1/2 at j+1
        KEEPdir(i,j+1,k,1,halfsten,pVcoeffs,prims,nfabfy,lclosures);
      }

      if (!markers(i,j-1,k,0)) { //the point to the down is fluid then compute fj-1/2 at j
        KEEPdir(i,j,k,1,halfsten,pVcoeffs,prims,nfabfy,lclosures);
      }

      if (!markers(i,j,k+1,0)) { //the point to the front is fluid then compute fk-1/2 at k
        KEEPdir(i,j,k+1,2,halfsten,pVcoeffs,prims,nfabfz,lclosures);
      }

      if (!markers(i,j,k-1,0)) { //the point to the back is fluid then compute fk-1/2 at k+1
        KEEPdir(i,j,k,2,halfsten,pVcoeffs,prims,nfabfy,lclosures);
      }

    });
  }
}
#endif
}
#endif