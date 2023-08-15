#ifndef Central_H_
#define Central_H_

#include <CNS.H>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>

using namespace amrex;

namespace Central {

  //2 * standard finite difference coefficients
#if !AMREX_USE_GPU
  extern GpuArray<Real,3> coeffs;
  extern GpuArray<Real,3> coeffs2;
#endif
  extern int order_keep;

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
  void KEEP(int i, int j, int k ,int halfsten, const GpuArray<Real,3>& coeffs, const auto& prims,const auto& nfabfx,const auto& nfabfy, const auto& nfabfz, Parm const& parm) {

    int i1, i2, j1, j2, k1, k2;
    Real rho1, ux1, uy1, uz1, ie1, p1;
    Real rho2, ux2, uy2, uz2, ie2, p2;
    Real massflx,ke;

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
        ie1  = parm.cv*prims(i1,j,k,QT);
        p1   = prims(i1,j,k,QPRES);

        i2   = i + m - l;
        rho2 = prims(i2,j,k,QRHO);
        ux2  = prims(i2,j,k,QU);
        uy2  = prims(i2,j,k,QV);
        uz2  = prims(i2,j,k,QW);
        ie2  = parm.cv*prims(i2,j,k,QT);
        p2   = prims(i2,j,k,QPRES);

        massflx = fgQuad(rho1,rho2,ux1,ux2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);

        nfabfx(i,j,k,URHO) += coeffs[l-1]*massflx;
        nfabfx(i,j,k,UMX ) += coeffs[l-1]*(fghCubic(rho1,rho2,ux1,ux2,ux1,ux2) +fDiv(p1,p2));
        nfabfx(i,j,k,UMY ) += coeffs[l-1]*fghCubic(rho1,rho2,ux1,ux2,uy1,uy2);
        nfabfx(i,j,k,UMZ ) += coeffs[l-1]*fghCubic(rho1,rho2,ux1,ux2,uz1,uz2);
        nfabfx(i,j,k,UET ) += coeffs[l-1]*(massflx*ke + fghCubic(rho1,rho2,ux1,ux2,ie1,ie2) + fgDiv(p1,p2,ux1,ux2));
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
        ie1  = parm.cv*prims(i,j1,k,QT);
        p1   = prims(i,j1,k,QPRES);

        j2   = j + m - l;
        rho2 = prims(i,j2,k,QRHO);
        ux2  = prims(i,j2,k,QU);
        uy2  = prims(i,j2,k,QV);
        uz2  = prims(i,j2,k,QW);
        ie2  = parm.cv*prims(i,j2,k,QT);
        p2   = prims(i,j2,k,QPRES);

        massflx = fgQuad(rho1,rho2,uy1,uy2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);
        nfabfy(i,j,k,URHO) += coeffs[l-1]*massflx;
        nfabfy(i,j,k,UMX ) += coeffs[l-1]*fghCubic(rho1,rho2,uy1,uy2,ux1,ux2);
        nfabfy(i,j,k,UMY ) += coeffs[l-1]*(fghCubic(rho1,rho2,uy1,uy2,uy1,uy2)+ fDiv(p1,p2));
        nfabfy(i,j,k,UMZ ) += coeffs[l-1]*fghCubic(rho1,rho2,uy1,uy2,uz1,uz2);
        nfabfy(i,j,k,UET ) += coeffs[l-1]*(massflx*ke + fghCubic(rho1,rho2,uy1,uy2,ie1,ie2)+ fgDiv(p1,p2,uy1,uy2));
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
        ie1  = parm.cv*prims(i,j,k1,QT);
        p1   = prims(i,j,k1,QPRES);

        k2   = k + m - l;
        rho2 = prims(i,j,k2,QRHO);
        ux2  = prims(i,j,k2,QU);
        uy2  = prims(i,j,k2,QV);
        uz2  = prims(i,j,k2,QW);
        ie2  = parm.cv*prims(i,j,k2,QT);
        p2   = prims(i,j,k2,QPRES);

        massflx = fgQuad(rho1,rho2,uz1,uz2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);
        nfabfz(i,j,k,URHO) += coeffs[l-1]*massflx;
        nfabfz(i,j,k,UMX)  += coeffs[l-1]*fghCubic(rho1,rho2,uz1,uz2,ux1,ux2);
        nfabfz(i,j,k,UMY)  += coeffs[l-1]*fghCubic(rho1,rho2,uz1,uz2,uy1,uy2);
        nfabfz(i,j,k,UMZ)  += coeffs[l-1]*(fghCubic(rho1,rho2,uz1,uz2,uz1,uz2) + fDiv(p1,p2));
        nfabfz(i,j,k,UET)  += coeffs[l-1]*(massflx*ke + fghCubic(rho1,rho2,uz1,uz2,ie1,ie2)+ fgDiv(p1,p2,uz1,uz2));
      }
    }

  }
  ///////////
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  void KEEPy(int i, int j, int k ,int halfsten, const GpuArray<Real,3> coeffs, const auto& prims, const auto& nfabfy, Parm const& parm) {

    GpuArray<Real,NCONS> flx; for (int n=0; n<NCONS; n++) {flx[n]=0.0;};
    int j1, j2;
    Real rho1, ux1, uy1, uz1, ie1, p1;
    Real rho2, ux2, uy2, uz2, ie2, p2;
    Real massflx,ke;

    // flux at j-1/2
    for (int l=1; l<=halfsten; l++) {
      for (int m=0; m<=l-1; m++) {
        j1 = j + m;
        rho1 = prims(i,j1,k,QRHO);
        ux1  = prims(i,j1,k,QU);
        uy1  = prims(i,j1,k,QV);
        uz1  = prims(i,j1,k,QW);
        ie1  = parm.cv*prims(i,j1,k,QT);
        p1   = prims(i,j1,k,QPRES);

        j2   = j + m - l;
        rho2 = prims(i,j2,k,QRHO);
        ux2  = prims(i,j2,k,QU);
        uy2  = prims(i,j2,k,QV);
        uz2  = prims(i,j2,k,QW);
        ie2  = parm.cv*prims(i,j2,k,QT);
        p2   = prims(i,j2,k,QPRES);

        massflx = fgQuad(rho1,rho2,uy1,uy2);
        ke      = 0.5_rt*(ux1*ux2 + uy1*uy2 + uz1*uz2);
        flx[URHO] = flx[URHO] + coeffs[l-1]*massflx;
        flx[UMX]  = flx[UMX]  + coeffs[l-1]*fghCubic(rho1,rho2,uy1,uy2,ux1,ux2);
        flx[UMY]  = flx[UMY]  + coeffs[l-1]*(fghCubic(rho1,rho2,uy1,uy2,uy1,uy2) + fDiv(p1,p2));
        flx[UMZ]  = flx[UMZ]  + coeffs[l-1]*fghCubic(rho1,rho2,uy1,uy2,uz1,uz2);
        flx[UET]  = flx[UET]  + coeffs[l-1]*(massflx*ke + fghCubic(rho1,rho2,uy1,uy2,ie1,ie2)+ fgDiv(p1,p2,uy1,uy2));
      }
    }
    for (int n=0; n<NCONS; n++) {nfabfy(i,j,k,n) = flx[n];}
  
  }


  inline void FluxKEEP(MultiFab& statemf,  MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {

    Parm& lparm = *CNS::d_parm;
    int halfsten = order_keep/2;
  // TODO: define coeffs outside of this routine and only once
#if AMREX_USE_GPU
  GpuArray<Real,3> coeffs;
  if (order_keep==6) {
    coeffs[0]=Real(6.0)/4; coeffs[1]=Real(-6.0)/20; coeffs[2]=Real(2.0)/60; 
  }
  else if (order_keep==4) { 
    coeffs[0]=Real(4.0)/3; coeffs[1]=Real(-2.0)/12; coeffs[2]=0.0;
    }
  else { coeffs[0]=Real(1.0);coeffs[1]=0.0; coeffs[2]=0.0; }
#endif
    ///////////////
    // GpuArray<Real,3>* coeffs; 
    // coeffs = (GpuArray<Real,3>*)The_Arena()->alloc(sizeof(GpuArray<Real,3>));
    // (*coeffs)[0]=Real(4.0)/3;
    // (*coeffs)[1]=Real(-2.0)/12;
    // (*coeffs)[2]=0.0;

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
        KEEP(i,j,k,halfsten,coeffs,prims,nfabfx,nfabfy,nfabfz,lparm);
      });
    }

    // The_Arena()->free(coeffs);
  }

  // TODO: In post_init select the boundaries and planes to reduce order/treat specially -- function to generate box arrays of planes with associated order of numericalscheme to be applied.
  inline void Flux_2nd_Order_KEEP(Geometry& geom, MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {

    BCRec& l_phys_bc = *CNS::d_phys_bc;
    Parm& lparm = *CNS::d_parm;
#if AMREX_USE_GPU
    GpuArray<Real,3> coeffs2;
    coeffs2[0]=Real(1.0);
    coeffs2[1]=0.0;
    coeffs2[2]=0.0;
#endif
    /////////////// 

    // GpuArray<Real,3>* coeffs2; 
    // coeffs2 = (GpuArray<Real,3>*)The_Arena()->alloc(sizeof(GpuArray<Real,3>));
    // (*coeffs2)[0]=Real(1.0);
    // (*coeffs2)[1]=0.0;
    // (*coeffs2)[2]=0.0;

    for (MFIter mfi(primsmf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bx      = mfi.tilebox();
      const Box& bxnodal = mfi.grownnodaltilebox(-1,0);
      auto const& prims  = primsmf.array(mfi);
      auto const& nfabfy = numflxmf[1].array(mfi);

      // Order reduction at ymin wall boundary
      if (l_phys_bc.lo(1)==6) {
        if (geom.Domain().smallEnd(1)==bx.smallEnd(1)) {
          int jbndry = bxnodal.smallEnd(1);
          IntVect small,big;
          small.setVal(0,bxnodal.smallEnd(0));
          small.setVal(1,jbndry);
          small.setVal(2,bxnodal.smallEnd(2));

          big.setVal(0,bxnodal.bigEnd(0));
          big.setVal(1,jbndry);
          big.setVal(2,bxnodal.bigEnd(2));
          Box bxboundary(small,big);

          ParallelFor(bxboundary, 
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            KEEPy(i,j,k,1,coeffs2,prims,nfabfy,lparm);
            // printf("2nd ord %d %d %d \n",i,j,k);
          });
        }
      }
    
    // Order reduction at ymax wall boundary
    if (l_phys_bc.hi(1)==6) {
      if (geom.Domain().bigEnd(1)==bx.bigEnd(1)) {
          int jbndry = bxnodal.bigEnd(1);
          IntVect small,big;
          small.setVal(0,bxnodal.smallEnd(0));
          small.setVal(1,jbndry);
          small.setVal(2,bxnodal.smallEnd(2));

          big.setVal(0,bxnodal.bigEnd(0));
          big.setVal(1,jbndry);
          big.setVal(2,bxnodal.bigEnd(2));
          Box bxboundary(small,big);

          ParallelFor(bxboundary, 
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            KEEPy(i,j,k,1,coeffs2,prims,nfabfy,lparm);
            // printf("2nd ord %d %d %d \n",i,j,k);
          });
        }
      }
    }

    // The_Arena()->free(coeffs2);
  }

}

#endif