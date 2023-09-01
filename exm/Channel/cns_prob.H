#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_parm.H"
#include "cns_prob_parm.H"
#include "CNS_index_macros.H"

using namespace amrex;

AMREX_GPU_DEVICE inline
void prob_initdata (int i, int j, int k, Array4<Real> const& state, GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm) {
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();

  Real pi = 3.14159265358979323846_rt;
  Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
  Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
  Real z = prob_lo[2] + (k+Real(0.5))*dx[2];

  Real lx = prob_hi[0] - prob_lo[0];
  Real ly = prob_hi[1] - prob_lo[1];
  Real lz = prob_hi[2] - prob_lo[2];

  if (y > prob_parm.h) {y = 2*prob_parm.h-y;}

  // Laminar velocity and constant temperature
  // Real ux = 1.5*prob_parm.ub*(1 - (1/std::pow(prob_parm.h,2))*std::pow((y-prob_parm.h),2));
  // Real ux = 0.0_rt;

  // Real T = prob_parm.Tw+0.5*prob_parm.Tw*(1 - (1/std::pow(prob_parm.h,2))*std::pow((y-prob_parm.h),2));
  // Real T = prob_parm.Tw;

  // Turbulent profiles
  Real T = 677.04110513 + 255.46817*pow(y,0.5) - 176.38967315*exp(-3081.95957181*y)*1.025;
  Real ux = 581.54524091 + 2687.84321585*pow(y,0.5) - 593.73493364* exp(-2008.87100276*y);

  int nx = 1;
  int ny = 2;
  int nz = 3;

  // srand(10);
  // Print() << (Real)rand()/ RAND_MAX << std::endl;
  // amrex::RandomEngine re;
// Real RandomNormal (Real mean, Real stddev, RandomEngine const& random_engine)
  // Real rand_num = 1.0;//amrex::RandomNormal(0.0, 0.0,re);
#if AMREX_USE_GPU
  Real rand_num = 0.5;
#else
  Real rand_num = (Real)rand()/ RAND_MAX;
#endif

  Real A       = 100*2.5;
  Real u_prime = rand_num*A*cos(2*pi*x*nx/lx)*sin(pi*y*ny/ly)*cos(2*pi*z*nz/lz);
  Real v_prime = rand_num*-A*sin(2*pi*x*nx/lx)*cos(pi*y*ny/ly)*sin(2*pi*z*nz/lz);
  Real w_prime = rand_num*-A*sin(2*pi*x*nx/lx)*cos(pi*y*ny/ly)*sin(2*pi*z*nz/lz);

  // Print() << u_prime << " " << v_prime << " " << w_prime << std::endl;

  Real rho =  prob_parm.Pw/(parm.Rspec*T);
  state(i,j,k,URHO ) = rho;
  state(i,j,k,UMX  ) = rho*(ux+u_prime);
  state(i,j,k,UMY  ) = rho*v_prime;
  state(i,j,k,UMZ  ) = rho*w_prime;
  state(i,j,k,UET  ) = rho*parm.cv*T + Real(0.5)*rho*(ux*ux + u_prime*u_prime + v_prime*v_prime + w_prime*w_prime);
}

inline void user_tagging(TagBoxArray& tags, const MultiFab &sdata, Geometry const& geom, int level) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(tags,TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto const& tagfab = tags.array(mfi);
    // auto const& sdf = sdata.array(mfi); // state data fab (sdf)
    // int idx = 0; // density index
    // const Real* dx      = geom.CellSize();
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
      // Real drhox = Math::abs(sdf(i+1,j,k,idx) - sdf(i-1,j,k,idx))/sdf(i,j,k,idx);
      tagfab(i,j,k) = (j==geom.Domain().smallEnd(1) || j==geom.Domain().bigEnd(1)) ;
      // tagfab(i,j,k) = j == 0 || j==127 ;
    });
  }
}


AMREX_GPU_DEVICE inline
void user_source(int i, int j, int k, const auto& state, const auto& rhs, const ProbParm& lprobparm,Parm const& parms, auto const dx) {

  // Print() << lprobparm.fx << std::endl;
  rhs(i,j,k,UMX) +=  state(i,j,k,URHO)*lprobparm.fx;
  rhs(i,j,k,UET) +=  state(i,j,k,UMX )*lprobparm.fx;

}

#endif
