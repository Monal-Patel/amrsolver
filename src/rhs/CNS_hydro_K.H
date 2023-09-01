#ifndef CNS_HYDRO_K_H_
#define CNS_HYDRO_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

using namespace amrex;

// prim = primitive variables
// cons = conserved variables
// char = characteristic variables
// flux = flux variables


// pointer function
// 

// Declaring
// void (*EulerFlux) (parameter type, ....); 

// Referencing
// EulerFlux=function_name;

// Dereferencing
// data_type x=*FuncPtr;


AMREX_GPU_DEVICE inline void cons2prim (int i, int j, int k, const Array4<const Real>& u, Array4<Real> const& q, const Parm& parm) noexcept
{
    Real rho = u(i,j,k,URHO);
    // Print() << "cons2prim"<< i << j << k << rho << std::endl;
    // rho = max(1e-40,rho);
    Real rhoinv = Real(1.0)/rho;
    Real ux = u(i,j,k,UMX)*rhoinv;
    Real uy = u(i,j,k,UMY)*rhoinv;
    Real uz = u(i,j,k,UMZ)*rhoinv;
    Real rhoke = Real(0.5)*rho*(ux*ux + uy*uy + uz*uz);
    Real rhoei = (u(i,j,k,UET) - rhoke);
    Real p = (parm.eos_gamma-Real(1.0))*rhoei;

    q(i,j,k,QRHO)  = rho;
    q(i,j,k,QU)    = ux;
    q(i,j,k,QV)    = uy;
    q(i,j,k,QW)    = uz;
    q(i,j,k,QPRES) = p;
    q(i,j,k,QT) = p/(rho*parm.Rspec);
}



// Discontinuity sensor
AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
Real disconSensor(Real pp, Real pl, Real pr) {
    Real pjst = pr + 2.0_rt*pp  + pl;
    Real ptvd = std::abs(pr-pp) + std::abs(pp -pl);
    return std::abs(2.0_rt* (pr -2.0_rt*pp + pl)/(pjst + ptvd + Real(1.0e-40)));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
void ComputeSensorLambda(int i, int j, int k, const auto& prims, const auto& lambda, const auto& sen, Parm const& parm) {

  // Real pp=prims(i,j,k,QPRES)*prims(i,j,k,QRHO);
  // sen(i,j,k,0) = disconSensor(pp,prims(i-1,j,k,QPRES)*prims(i-1,j,k,QRHO),prims(i+1,j,k,QPRES)*prims(i+1,j,k,QRHO));
  // sen(i,j,k,1) = disconSensor(pp,prims(i,j-1,k,QPRES)*prims(i,j-1,k,QRHO),prims(i,j+1,k,QPRES)*prims(i,j+1,k,QRHO));
  // sen(i,j,k,2) = disconSensor(pp,prims(i,j,k-1,QPRES)*prims(i,j,k-1,QRHO),prims(i,j,k+1,QPRES)*prims(i,j,k+1,QRHO));

  // Real pp=prims(i,j,k,QRHO);
  // sen(i,j,k,0) = disconSensor(pp,prims(i-1,j,k,QRHO),prims(i+1,j,k,QRHO));
  // sen(i,j,k,1) = disconSensor(pp,prims(i,j-1,k,QRHO),prims(i,j+1,k,QRHO));
  // sen(i,j,k,2) = disconSensor(pp,prims(i,j,k-1,QRHO),prims(i,j,k+1,QRHO));

  Real pp=prims(i,j,k,QPRES);
  sen(i,j,k,0) = disconSensor(pp,prims(i-1,j,k,QPRES),prims(i+1,j,k,QPRES));
  sen(i,j,k,1) = disconSensor(pp,prims(i,j-1,k,QPRES),prims(i,j+1,k,QPRES));
  sen(i,j,k,2) = disconSensor(pp,prims(i,j,k-1,QPRES),prims(i,j,k+1,QPRES));

  Real ux = prims(i,j,k,QU); 
  Real uy = prims(i,j,k,QV);
  Real uz = prims(i,j,k,QW);
  Real cs = sqrt(parm.eos_gamma*prims(i,j,k,QPRES)/prims(i,j,k,QRHO)); 
  lambda(i,j,k,0) = std::abs(ux)+cs; //max(std::abs(ux+cs),std::abs(ux-cs)); 
  lambda(i,j,k,1) = std::abs(uy)+cs;//max(std::abs(uy+cs),std::abs(uy-cs)); 
  lambda(i,j,k,2) = std::abs(uz)+cs;//max(std::abs(uz+cs),std::abs(uz-cs)); 
  // lambda(i,j,k,0) = max(std::abs(ux+cs),std::abs(ux-cs)); 
  // lambda(i,j,k,1) = max(std::abs(uy+cs),std::abs(uy-cs)); 
  // lambda(i,j,k,2) = max(std::abs(uz+cs),std::abs(uz-cs)); 

}

// calculates dissipative flux at i-1/2, j-1/2 and k-1/2
AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
void JSTflux(int i, int j, int k, int n, const auto& lambda , const auto& sensor, const auto& cons,const auto& nfabfx,const auto& nfabfy, const auto& nfabfz, Parm const& parm) {

  Real dw,sen,rr,fdamp;
  Real u_ijk = cons(i,j,k,n);

  // x-dir
  dw     = (u_ijk - cons(i-1,j,k,n));
  sen    = parm.Cshock*max(sensor(i-1,j,k,0),sensor(i,j,k,0));
  rr     = max(lambda(i-1,j,k,0),lambda(i,j,k,0));
  fdamp  = cons(i+1,j,k,n) - 3.0*cons(i,j,k,n) + 3.0*cons(i-1,j,k,n) - cons(i-2,j,k,n);
  nfabfx(i,j,k,n) -= (sen*dw - max(0.0,parm.Cdamp - sen)*fdamp)*rr;

  // y-dir
  dw     = (u_ijk - cons(i,j-1,k,n));
  sen    = parm.Cshock*max(sensor(i,j-1,k,0),sensor(i,j,k,0));
  rr     = max(lambda(i,j-1,k,0),lambda(i,j,k,0));
  fdamp  = cons(i,j+1,k,n) - 3.0*cons(i,j,k,n) + 3.0*cons(i,j-1,k,n) - cons(i,j-2,k,n);
  nfabfy(i,j,k,n) -= (sen*dw - max(0.0,parm.Cdamp - sen)*fdamp)*rr;

  // z-dir
  dw     = (u_ijk - cons(i,j,k-1,n));
  sen    = parm.Cshock*max(sensor(i,j,k-1,0),sensor(i,j,k,0));
  rr     = max(lambda(i,j,k-1,0),lambda(i,j,k,0));
  fdamp  = cons(i,j,k-1,n) - 3.0*cons(i,j,k,n) + 3.0*cons(i,j,k-1,n) - cons(i,j,k-1,n);
  nfabfz(i,j,k,n) -= (sen*dw - max(0.0,parm.Cdamp - sen)*fdamp)*rr;
}


// Viscous fluxes at cell centers
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void viscfluxes (int i, int j, int k, auto const& prims, auto const& fx, auto const& fy, auto const& fz, GpuArray<Real,AMREX_SPACEDIM> const& dxinv, Parm const& parms) noexcept
{
    // 2nd order accurate central difference of primitive vars /////////////////
    // x direction
    Real ux   =  prims(i,j,k,QU);
    Real dudx = (prims(i+1,j,k,QU) - prims(i-1,j,k,QU))*0.5_rt*dxinv[0];
    Real dvdx = (prims(i+1,j,k,QV) - prims(i-1,j,k,QV))*0.5_rt*dxinv[0];
    Real dwdx = (prims(i+1,j,k,QW) - prims(i-1,j,k,QW))*0.5_rt*dxinv[0];
    Real dTdx = (prims(i+1,j,k,QT) - prims(i-1,j,k,QT))*0.5_rt*dxinv[0];

    // y direction
    Real uy   =  prims(i,j,k,QV);
    Real dudy = (prims(i,j+1,k,QU) - prims(i,j-1,k,QU))*0.5_rt*dxinv[1];
    Real dvdy = (prims(i,j+1,k,QV) - prims(i,j-1,k,QV))*0.5_rt*dxinv[1];
    Real dwdy = (prims(i,j+1,k,QW) - prims(i,j-1,k,QW))*0.5_rt*dxinv[1];
    Real dTdy = (prims(i,j+1,k,QT) - prims(i,j-1,k,QT))*0.5_rt*dxinv[1];

    // z direction
    Real uz   =  prims(i,j,k,QW);
    Real dudz = (prims(i,j,k+1,QU) - prims(i,j,k-1,QU))*0.5_rt*dxinv[2];
    Real dvdz = (prims(i,j,k+1,QV) - prims(i,j,k-1,QV))*0.5_rt*dxinv[2];
    Real dwdz = (prims(i,j,k+1,QW) - prims(i,j,k-1,QW))*0.5_rt*dxinv[2];
    Real dTdz = (prims(i,j,k+1,QT) - prims(i,j,k-1,QT))*0.5_rt*dxinv[2];

    // divergence
    Real div  = dudx + dvdy + dwdz;

    // constants
    Real mu    = parms.ViscositySutherland(prims(i,j,k,QT));
    Real lambda= parms.ConductivityConstPr(prims(i,j,k,QT));
    Real r1_3  = Real(1.0)/Real(3.0);

    // viscous fluxes
    Real tauxx = Real(2.0)*mu*(dudx - r1_3*div);
    Real tauxy = mu*(dudy + dvdx);
    Real tauxz = mu*(dudz + dwdx);

    // tauxy = tauyx
    Real tauyy = Real(2.0)*mu*(dvdy - r1_3*div);
    Real tauyz = mu*(dvdz + dwdy);

    // tauzx = tauxz;
    // tauzy = tauyz;
    Real tauzz = Real(2.0)*mu*(dwdz - r1_3*div);

    // assemble fluxes on LHS
    fx(i,j,k,URHO)= Real(0.0);
    fx(i,j,k,UMX) = -tauxx;
    fx(i,j,k,UMY) = -tauxy;
    fx(i,j,k,UMZ) = -tauxz;
    fx(i,j,k,UET) = -lambda*dTdx - tauxx*ux - tauxy*uy - tauxz*uz ;

    fy(i,j,k,URHO)= Real(0.0);
    fy(i,j,k,UMX) = -tauxy;
    fy(i,j,k,UMY) = -tauyy;
    fy(i,j,k,UMZ) = -tauyz;
    fy(i,j,k,UET) = -lambda*dTdy - tauxy*ux - tauyy*uy - tauyz*uz;

    fz(i,j,k,URHO)= Real(0.0);
    fz(i,j,k,UMX) = -tauxz;
    fz(i,j,k,UMY) = -tauyz;
    fz(i,j,k,UMZ) = -tauzz;
    fz(i,j,k,UET) = -lambda*dTdz - tauxz*ux - tauyz*uy - tauzz*uz;
}


// Viscous fluxes at cell centers
// TODO: remove if statement and generalise to other directions by passing one-sided derivative coefficients
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void viscfluxes_wall (int i, int j, int k, int loc, auto const& prims, auto const& fx, auto const& fy, auto const& fz, GpuArray<Real,AMREX_SPACEDIM> const& dxinv, Parm const& parms) noexcept
{
    // 2nd order accurate central difference of primitive vars /////////////////
    // x direction
    Real ux   =  prims(i,j,k,QU);
    Real dudx = (prims(i+1,j,k,QU) - prims(i-1,j,k,QU))*0.5_rt*dxinv[0];
    Real dvdx = (prims(i+1,j,k,QV) - prims(i-1,j,k,QV))*0.5_rt*dxinv[0];
    Real dwdx = (prims(i+1,j,k,QW) - prims(i-1,j,k,QW))*0.5_rt*dxinv[0];
    Real dTdx = (prims(i+1,j,k,QT) - prims(i-1,j,k,QT))*0.5_rt*dxinv[0];

    // y direction
    Real uy   =  prims(i,j,k,QV);
    Real dudy, dvdy, dwdy, dTdy;
    // bottom boundary
    if (loc==0) {
    dudy = (-prims(i,j+2,k,QU) + 4*prims(i,j+1,k,QU) - 3*prims(i,j,k,QU))*0.5_rt*dxinv[1];
    dvdy = (-prims(i,j+2,k,QV) + 4*prims(i,j+1,k,QV) - 3*prims(i,j,k,QV))*0.5_rt*dxinv[1];
    dwdy = (-prims(i,j+2,k,QW) + 4*prims(i,j+1,k,QW) - 3*prims(i,j,k,QW))*0.5_rt*dxinv[1];
    dTdy = (-prims(i,j+2,k,QT) + 4*prims(i,j+1,k,QT) - 3*prims(i,j,k,QT))*0.5_rt*dxinv[1];}
    // top boundary
    else if (loc==1) {
    dudy = (3*prims(i,j,k,QU) - 4*prims(i,j-1,k,QU) + prims(i,j-2,k,QU))*0.5_rt*dxinv[1];
    dvdy = (3*prims(i,j,k,QV) - 4*prims(i,j-1,k,QV) + prims(i,j-2,k,QV))*0.5_rt*dxinv[1];
    dwdy = (3*prims(i,j,k,QW) - 4*prims(i,j-1,k,QW) + prims(i,j-2,k,QW))*0.5_rt*dxinv[1];
    dTdy = (3*prims(i,j,k,QT) - 4*prims(i,j-1,k,QT) + prims(i,j-2,k,QT))*0.5_rt*dxinv[1];
    }

    // z direction
    Real uz   =  prims(i,j,k,QW);
    Real dudz = (prims(i,j,k+1,QU) - prims(i,j,k-1,QU))*0.5_rt*dxinv[2];
    Real dvdz = (prims(i,j,k+1,QV) - prims(i,j,k-1,QV))*0.5_rt*dxinv[2];
    Real dwdz = (prims(i,j,k+1,QW) - prims(i,j,k-1,QW))*0.5_rt*dxinv[2];
    Real dTdz = (prims(i,j,k+1,QT) - prims(i,j,k-1,QT))*0.5_rt*dxinv[2];

    // divergence
    Real div  = dudx + dvdy + dwdz;

    // constants
    Real mu    = parms.ViscositySutherland(prims(i,j,k,QT));
    Real lambda= parms.ConductivityConstPr(prims(i,j,k,QT));
    Real r1_3  = Real(1.0)/Real(3.0);

    // viscous fluxes
    Real tauxx = Real(2.0)*mu*(dudx - r1_3*div);
    Real tauxy = mu*(dudy + dvdx);
    Real tauxz = mu*(dudz + dwdx);

    // tauxy = tauyx
    Real tauyy = Real(2.0)*mu*(dvdy - r1_3*div);
    Real tauyz = mu*(dvdz + dwdy);

    // tauzx = tauxz;
    // tauzy = tauyz;
    Real tauzz = Real(2.0)*mu*(dwdz - r1_3*div);

    // assemble fluxes on LHS
    fx(i,j,k,URHO)= Real(0.0);
    fx(i,j,k,UMX) = -tauxx;
    fx(i,j,k,UMY) = -tauxy;
    fx(i,j,k,UMZ) = -tauxz;
    fx(i,j,k,UET) = -lambda*dTdx - tauxx*ux - tauxy*uy - tauxz*uz ;

    fy(i,j,k,URHO)= Real(0.0);
    fy(i,j,k,UMX) = -tauxy;
    fy(i,j,k,UMY) = -tauyy;
    fy(i,j,k,UMZ) = -tauyz;
    fy(i,j,k,UET) = -lambda*dTdy - tauxy*ux - tauyy*uy - tauyz*uz;

    fz(i,j,k,URHO)= Real(0.0);
    fz(i,j,k,UMX) = -tauxz;
    fz(i,j,k,UMY) = -tauyz;
    fz(i,j,k,UMZ) = -tauzz;
    fz(i,j,k,UET) = -lambda*dTdz - tauxz*ux - tauyz*uy - tauzz*uz;
}

// 2nd order accurate interpolation at i-1/2,j-1/2,k-1/2
AMREX_GPU_DEVICE inline 
void visc_numericalfluxes(int i, int j, int k, int n, const auto& pfx, const auto& pfy, const auto& pfz, const auto& nfx, const auto& nfy, const auto& nfz) {

nfx(i,j,k,n) += Real(0.5)*(pfx(i-1,j,k,n) + pfx(i,j,k,n));
nfy(i,j,k,n) += Real(0.5)*(pfy(i,j-1,k,n) + pfy(i,j,k,n));
nfz(i,j,k,n) += Real(0.5)*(pfz(i,j,k-1,n) + pfz(i,j,k,n));

}

#endif