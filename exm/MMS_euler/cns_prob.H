#ifndef CNS_PROB_H
#define CNS_PROB_H

#include <cmath>

#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "CNS.H"
#include "CNS_index_macros.H"
#include "cns_prob_parm.H"

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_initdata (
  int i, int j, int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  Parm const& parm,
  ProbParm const& pparm)
{
  // Geometry
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();
  const Real x = prob_lo[0] + (i + 0.5) * dx[0];
  const Real y = prob_lo[1] + (j + 0.5) * dx[1];
  const Real z = prob_lo[2] + (k + 0.5) * dx[2];

  // MMS functions

  Real u[3],p,T,rho,eint;
  u[0] = pparm.fu(x,y,z);
  u[1] = pparm.fv(x,y,z);
  u[2] = pparm.fw(x,y,z);
  p = pparm.fp(x,y,z);
  T = pparm.ft(x,y,z);
  rho  = p/(parm.Rspec*T); 
  eint = parm.cv*T;

  // Set the state
  state(i, j, k, URHO) = rho;
  state(i, j, k, UMX)  = rho * u[0];
  state(i, j, k, UMY)  = rho * u[1];
  state(i, j, k, UMZ)  = rho * u[2];
  state(i, j, k, UET)  = rho * (eint + Real(0.5) * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]));
}

inline void user_tagging(TagBoxArray& tags, MultiFab &sdata, int level) {
}


AMREX_GPU_DEVICE inline
void user_source(int i, int j, int k, const auto& state, const auto& rhs, const ProbParm& pparm,Parm const& parms, auto const dx) {


  const Real x = (i + 0.5_rt) * dx[0];
  const Real y = (j + 0.5_rt) * dx[1];
  const Real z = (k + 0.5_rt) * dx[2];

  Real rho = 1/parms.Rspec;

  // rhs(i,j,k,URHO) =rhs(i,j,k,URHO) + rho*(pparm.dudx(x,y,z) + pparm.dvdy(x,y,z) + pparm.dwdz(x,y,z));

  Real error = rhs(i,j,k,URHO) + rho*(pparm.dudx(x,y,z) + pparm.dvdy(x,y,z) + pparm.dwdz(x,y,z));
  rhs(i,j,k,URHO) = error;

  // rho (duu/dx + duv/dy + duw/dz) assuming rho=constant
  error = rhs(i,j,k,UMX) + rho*( 2*pparm.fu(x,y,z)*pparm.dudx(x,y,z) 
                 + pparm.fv(x,y,z)*pparm.dudy(x,y,z) + pparm.fu(x,y,z)*pparm.dvdy(x,y,z)
                 + pparm.fw(x,y,z)*pparm.dudz(x,y,z) + pparm.fu(x,y,z)*pparm.dwdz(x,y,z)) 
                 + pparm.dpdx(x,y,z);
  rhs(i,j,k,UMX) = error;

  // rho (dvu/dx + dvv/dy + dvw/dz) assuming rho=constant
  error = rhs(i,j,k,UMY) + rho*( pparm.fu(x,y,z)*pparm.dvdx(x,y,z) + pparm.fv(x,y,z)*pparm.dudx(x,y,z) 
                 + 2*pparm.fv(x,y,z)*pparm.dvdy(x,y,z) 
                 + pparm.fw(x,y,z)*pparm.dvdz(x,y,z) + pparm.fv(x,y,z)*pparm.dwdz(x,y,z)) 
                 + pparm.dpdy(x,y,z);
  rhs(i,j,k,UMY) = error;

  // rho (dwu/dx + dwv/dy + dww/dz) assuming rho=constant
  error = rhs(i,j,k,UMZ) + rho*(pparm.fu(x,y,z)*pparm.dwdx(x,y,z) + pparm.fw(x,y,z)*pparm.dudx(x,y,z) 
        + pparm.fv(x,y,z)*pparm.dwdy(x,y,z) + pparm.fw(x,y,z)*pparm.dvdy(x,y,z)
        + 2*pparm.fw(x,y,z)*pparm.dwdz(x,y,z))
        + pparm.dpdz(x,y,z);
  rhs(i,j,k,UMZ) = error;

  // d (rho u ht) = d(rho u (et + p/rho)) = d(rho u et + Pu) = rho d(u et + Pu/rho) = rho [ d(u et) + 1/rho d(Pu) ] = rho [ et d(u) + u d(et) + (1/rho) ( u d(P) + P d(u) )]

  // et = e + 1/2 (u^2 + v^2 + w^2) = cv T + 1/2 (u^2 + v^2 + w^2)
  Real et = parms.cv*pparm.ft(x,y,z) + 0.5_rt*(pparm.fu(x,y,z)*pparm.fu(x,y,z) + pparm.fv(x,y,z)*pparm.fv(x,y,z) + pparm.fw(x,y,z)*pparm.fw(x,y,z));

  // x-direction: rho [ et d(u)/dx + u d(et)/dx + (1/rho) ( u d(P)dx + P d(u)dx )]
  // d(et)/dx = cv dT/dx + 2*0.5(udu/dx + vdv/dx + wdw/dx)
  Real det = parms.cv*pparm.dTdx(x,y,z) + (pparm.fu(x,y,z)*pparm.dudx(x,y,z) + pparm.fv(x,y,z)*pparm.dvdx(x,y,z) + pparm.fw(x,y,z)*pparm.dwdx(x,y,z));

  error = rho*( et*pparm.dudx(x,y,z) +  pparm.fu(x,y,z)*det + (1/rho)*(pparm.fu(x,y,z)*pparm.dpdx(x,y,z) + pparm.fp(x,y,z)*pparm.dudx(x,y,z)) );

  // y-direction: rho [ et d(v) + v d(et) + (1/rho) ( v d(P) + P d(v) )]
  det = parms.cv*pparm.dTdy(x,y,z) + (pparm.fu(x,y,z)*pparm.dudy(x,y,z) + pparm.fv(x,y,z)*pparm.dvdy(x,y,z) + pparm.fw(x,y,z)*pparm.dwdy(x,y,z));

  error += rho*( et*pparm.dvdy(x,y,z) +  pparm.fv(x,y,z)*det + (1/rho)*(pparm.fv(x,y,z)*pparm.dpdy(x,y,z) + pparm.fp(x,y,z)*pparm.dvdy(x,y,z)) );

  // z-direction: rho [ et d(w) + w d(et) + (1/rho) ( w d(P) + P d(w) )]
  det = parms.cv*pparm.dTdz(x,y,z) + (pparm.fu(x,y,z)*pparm.dudz(x,y,z) + pparm.fv(x,y,z)*pparm.dvdz(x,y,z) + pparm.fw(x,y,z)*pparm.dwdz(x,y,z));

  error += rho*( et*pparm.dwdz(x,y,z) +  pparm.fw(x,y,z)*det + (1/rho)*(pparm.fw(x,y,z)*pparm.dpdz(x,y,z) + pparm.fp(x,y,z)*pparm.dwdz(x,y,z)) );

  // Print() << x << " " << y << " " << z << " " << rhs(i,j,k,UET)/2 << " " << error << std::endl;
  // exit(0);

  error = error + rhs(i,j,k,UET);

  rhs(i,j,k,UET) = error;

}

#endif
