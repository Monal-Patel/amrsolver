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

///////////////////////////

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


//////////////////////////
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

// Fluxes
//------------------------------------------------------------------------------
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
    Real mu    = parms.visc(prims(i,j,k,QT));
    Real lambda= parms.cond(prims(i,j,k,QT));
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
