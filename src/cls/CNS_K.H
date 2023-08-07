#ifndef CNS_K_H_
#define CNS_K_H_

#include "CNS_index_macros.H"
#include <AMReX_FArrayBox.H>
#include <limits>
#include <cmath>

#include "CNS_parm.H"

// TODO: Pass primsmf here to avoid computing primitive variables again.
AMREX_GPU_HOST_DEVICE 
inline amrex::Real cns_estdt (amrex::Box const& bx, amrex::Array4<Real const>   const& state, amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx, Parm const& parm) noexcept
{
    using amrex::Real;

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
    Real dt = std::numeric_limits<Real>::max();
#else
    Real dt = Real(1.e37);
#endif

    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                Real rho = state(i,j,k,URHO);
                Real mx  = state(i,j,k,UMX);
                Real my  = state(i,j,k,UMY);
                Real mz  = state(i,j,k,UMY);
                Real rhoinv = Real(1.0)/amrex::max(rho,parm.smallr);
                Real vx = mx*rhoinv;
                Real vy = my*rhoinv;
                Real vz = mz*rhoinv;
                Real ke = Real(0.5)*rho*(vx*vx + vy*vy + vz*vz);
                Real rhoei = (state(i,j,k,UET) - ke);
                Real p  = (parm.eos_gamma-Real(1.0))*rhoei;
                Real cs = std::sqrt(parm.eos_gamma*p*rhoinv);
                Real dtx = dx[0]/(amrex::Math::abs(vx)+cs);
                Real dty = dx[1]/(amrex::Math::abs(vy)+cs);
                Real dtz = dx[2]/(amrex::Math::abs(vz)+cs);
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
            }
        }
    }

    return dt;
}

AMREX_GPU_DEVICE inline void pointCFL (int i, int j, int k, Array2D<Real,0,2,0,2>& array, const Array4<const Real>& prims, const Parm& parm,const auto dx, Real dt) noexcept
{
  Real sos = std::sqrt(parm.eos_gamma*parm.Rspec*prims(i,j,k,QT));
  //
  for (int idir=0; idir<AMREX_SPACEDIM; idir++){
    array(idir,0) = max(array(idir,0), std::abs(prims(i,j,k,QU+idir) + sos)); // max(|ui + c|) in +ith direction
    array(idir,1) = max(array(idir,1),std::abs(prims(i,j,k,QU+idir) - sos)); // max(|ui - c|) in -ith direction
    array(idir,2) = max(array(idir,0),array(idir,1))*dt/dx[idir]; // CFL max in ith direction
    // Print() << "inside " << cfl_array(idir,0) << " " << cfl_array(idir,1) << " " << cfl_array(idir,2) << std::endl; 
  }
}


AMREX_GPU_DEVICE
inline
void
cns_compute_temperature (int i, int j, int k, amrex::Array4<amrex::Real> const& u,
                         Parm const& parm) noexcept
{
    using amrex::Real;

    Real rhoinv = Real(1.0)/u(i,j,k,URHO);
    Real mx = u(i,j,k,UMX);
    Real my = u(i,j,k,UMY);
    Real mz = u(i,j,k,UMZ);
    Real rhoeint = u(i,j,k,UET) - Real(0.5) * rhoinv * (mx*mx+my*my+mz*mz);
    // u(i,j,k,UTEMP) = rhoinv * rhoeint * (Real(1.0)/parm.cv);
}

#endif
