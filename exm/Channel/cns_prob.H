#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_parm.H"
#include "cns_prob_parm.H"
#include "CNS_index_macros.H"

using namespace amrex;

AMREX_GPU_DEVICE inline
void user_initdata (int i, int j, int k, Array4<Real> const& state, GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm) {
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();

  Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
  if (y > prob_parm.h) {y = 2*prob_parm.h-y;}
  // Real ux = 581.54524091 + 2687.84321585*pow(y,0.5) - 593.73493364* exp(-2008.87100276*y);
  Real ux = 1.5*prob_parm.ub*(1 - (1/std::pow(prob_parm.h,2))*std::pow((y-prob_parm.h),2));
  Real T = 677.04110513 + 255.46817*pow(y,0.5) - 176.38967315*exp(-3081.95957181*y);

  // printf("%f %f %f \n",y,ux,T);

  Real rho =  prob_parm.Pw/(parm.Rspec*T);
  state(i,j,k,URHO ) = rho;
  state(i,j,k,UMX  ) = rho*ux;
  state(i,j,k,UMY  ) = Real(0.0);
  state(i,j,k,UMZ  ) = Real(0.0);
  state(i,j,k,UET) = rho*parm.cv*T + Real(0.5)*rho*ux*ux;
}

inline void user_tagging(TagBoxArray& tags, MultiFab &sdata, int level) {
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


AMREX_GPU_DEVICE inline
void user_source(int i, int j, int k, const auto& state, const auto& rhs, const ProbParm& lprobparm) {

  rhs(i,j,k,UMX) +=  state(i,j,k,URHO)*lprobparm.fx;
  rhs(i,j,k,UET) +=  state(i,j,k,UMX )*lprobparm.fx;

}

#endif
