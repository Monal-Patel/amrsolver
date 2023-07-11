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
//         // Real rmpoc2 = q(i,j,k,QRHO) - q(i,j,k,QPRES)/q(i,j,k,QC)/q(i,j,k,QC); //rho minus p over c^2
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
void JSTflux(int i, int j, int k, int n, const auto& lambda , const auto& sensor, const auto& cons,const auto& nfabfx,const auto& nfabfy, const auto& nfabfz) {

  Real c2 = 1.0_rt;
  Real c4 = 0.1_rt;
  Real dw,sen,rr,fdamp;
  Real u_ijk = cons(i,j,k,n);

  // x-dir
  dw     = (u_ijk - cons(i-1,j,k,n));
  sen    = c2*max(sensor(i-1,j,k,0),sensor(i,j,k,0));
  rr     = max(lambda(i-1,j,k,0),lambda(i,j,k,0));
  fdamp  = cons(i+1,j,k,n) - 3.0*cons(i,j,k,n) + 3.0*cons(i-1,j,k,n) - cons(i-2,j,k,n);
  nfabfx(i,j,k,n) -= (sen*dw - max(0.0,c4 - sen)*fdamp)*rr;

  // y-dir
  dw     = (u_ijk - cons(i,j-1,k,n));
  sen    = c2*max(sensor(i,j-1,k,0),sensor(i,j,k,0));
  rr     = max(lambda(i,j-1,k,0),lambda(i,j,k,0));
  fdamp  = cons(i,j+1,k,n) - 3.0*cons(i,j,k,n) + 3.0*cons(i,j-1,k,n) - cons(i,j-2,k,n);
  nfabfy(i,j,k,n) -= (sen*dw - max(0.0,c4 - sen)*fdamp)*rr;

  // z-dir
  dw     = (u_ijk - cons(i,j,k-1,n));
  sen    = c2*max(sensor(i,j,k-1,0),sensor(i,j,k,0));
  rr     = max(lambda(i,j,k-1,0),lambda(i,j,k,0));
  fdamp  = cons(i,j,k-1,n) - 3.0*cons(i,j,k,n) + 3.0*cons(i,j,k-1,n) - cons(i,j,k-1,n);
  nfabfz(i,j,k,n) -= (sen*dw - max(0.0,c4 - sen)*fdamp)*rr;
}

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
void KEEP(int i, int j, int k ,int halfsten, auto const& coeffs, const auto& prims,const auto& nfabfx,const auto& nfabfy, const auto& nfabfz, Parm const& parm) {

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
void KEEPy(int i, int j, int k ,int halfsten, auto const& coeffs, const auto& prims, const auto& nfabfy, Parm const& parm) {

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
Real limiter (Real dlft, Real drgt) noexcept
{
    Real dcen = Real(0.5)*(dlft+drgt);
    Real dsgn = Math::copysign(Real(1.0), dcen);
    Real slop = Real(2.0) * min(Math::abs(dlft),Math::abs(drgt));
    Real dlim = (dlft*drgt >= Real(0.0)) ? slop : Real(0.0);
    return dsgn * min(dlim,Math::abs(dcen));
}

AMREX_GPU_DEVICE
inline
void
cns_slope_x (int i, int j, int k,
             Array4<Real> const& dq,
             Array4<Real const> const& q, Parm const& parm) noexcept
{
    
    Real cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k,QT)) + 1.e-40;
    Real dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/cspeed - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
    Real drgt = Real(0.5)*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/cspeed - Real(0.5)*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
    Real d0 = limiter(dlft, drgt);

    Real cs2 = cspeed*cspeed;
    dlft = (q(i,j,k,QRHO)-q(i-1,j,k,QRHO)) - (q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/cs2;
    drgt = (q(i+1,j,k,QRHO)-q(i,j,k,QRHO)) - (q(i+1,j,k,QPRES) - q(i,j,k,QPRES))/cs2;
    Real d1 = limiter(dlft, drgt);

    dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/cspeed + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
    drgt = Real(0.5)*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/cspeed + Real(0.5)*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
    Real d2 = limiter(dlft, drgt);

    dlft = q(i,j,k,QV) - q(i-1,j,k,QV);
    drgt = q(i+1,j,k,QV) - q(i,j,k,QV);
    Real d3 = limiter(dlft, drgt);

    dlft = q(i,j,k,QW) - q(i-1,j,k,QW);
    drgt = q(i+1,j,k,QW) - q(i,j,k,QW);
    Real d4 = limiter(dlft, drgt);

    dq(i,j,k,0) = d0;
    dq(i,j,k,1) = d1;
    dq(i,j,k,2) = d2;
    dq(i,j,k,3) = d3;
    dq(i,j,k,4) = d4;
}

AMREX_GPU_DEVICE
inline
void
cns_slope_y (int i, int j, int k,
             Array4<Real> const& dq,
             Array4<Real const> const& q, Parm const& parm) noexcept
{

    Real cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k,QT)) + 1.e-40;
    Real dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/cspeed - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
    Real drgt = Real(0.5)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/cspeed - Real(0.5)*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
    Real d0 = limiter(dlft, drgt);

    Real cs2 = cspeed*cspeed;
    dlft = (q(i,j,k,QRHO)-q(i,j-1,k,QRHO)) - (q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/cs2;
    drgt = (q(i,j+1,k,QRHO)-q(i,j,k,QRHO)) - (q(i,j+1,k,QPRES) - q(i,j,k,QPRES))/cs2;
    Real d1 = limiter(dlft, drgt);

    dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/cspeed + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
    drgt = Real(0.5)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/cspeed + Real(0.5)*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
    Real d2 = limiter(dlft, drgt);

    dlft = q(i,j,k,QU) - q(i,j-1,k,QU);
    drgt = q(i,j+1,k,QU) - q(i,j,k,QU);
    Real d3 = limiter(dlft, drgt);

    dlft = q(i,j,k,QW) - q(i,j-1,k,QW);
    drgt = q(i,j+1,k,QW) - q(i,j,k,QW);
    Real d4 = limiter(dlft, drgt);

    dq(i,j,k,0) = d0;
    dq(i,j,k,1) = d1;
    dq(i,j,k,2) = d2;
    dq(i,j,k,3) = d3;
    dq(i,j,k,4) = d4;
}

AMREX_GPU_DEVICE inline
void cns_slope_z (int i, int j, int k,
             Array4<Real> const& dq,
             Array4<Real const> const& q, Parm const& parm) noexcept
{
    
    Real cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k,QT)) + 1.e-40;
    Real dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/cspeed - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QW) - q(i,j,k-1,QW));
    Real drgt = Real(0.5)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/cspeed - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k+1,QW) - q(i,j,k,QW));
    Real d0 = limiter(dlft, drgt);

    Real cs2 = cspeed*cspeed ;
    dlft = (q(i,j,k,QRHO)-q(i,j,k-1,QRHO)) - (q(i,j,k,QPRES) - q(i,j,k-1,QPRES))/cs2;
    drgt = (q(i,j,k+1,QRHO)-q(i,j,k,QRHO)) - (q(i,j,k+1,QPRES) - q(i,j,k,QPRES))/cs2;
    Real d1 = limiter(dlft, drgt);

    dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/cspeed + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QW) - q(i,j,k-1,QW));
    drgt = Real(0.5)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/cspeed + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k+1,QW) - q(i,j,k,QW));
    Real d2 = limiter(dlft, drgt);

    dlft = q(i,j,k,QU) - q(i,j,k-1,QU);
    drgt = q(i,j,k+1,QU) - q(i,j,k,QU);
    Real d3 = limiter(dlft, drgt);

    dlft = q(i,j,k,QV) - q(i,j,k-1,QV);
    drgt = q(i,j,k+1,QV) - q(i,j,k,QV);
    Real d4 = limiter(dlft, drgt);

    dq(i,j,k,0) = d0;
    dq(i,j,k,1) = d1;
    dq(i,j,k,2) = d2;
    dq(i,j,k,3) = d3;
    dq(i,j,k,4) = d4;
}


AMREX_GPU_DEVICE inline
void riemann (const Real gamma, const Real smallp, const Real /*smallr*/,
         const Real rl, const Real ul, const Real pl,
         const Real ut1l, const Real ut2l,
         const Real rr, const Real ur, const Real pr,
         const Real ut1r, const Real ut2r,
         Real& flxrho, Real& flxu, Real& flxut,
         Real& flxutt, Real& flxe) noexcept
{
    

    constexpr Real weakwv = Real(1.e-3);
    constexpr Real small = Real(1.e-6);

    Real clsql = gamma*pl*rl;
    Real clsqr = gamma*pr*rr;
    Real wl = std::sqrt(clsql);
    Real wr = std::sqrt(clsqr);
    Real cleft = wl/rl;
    Real cright = wr/rr;
    Real ccsmall = small*(cleft+cright);

    Real pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = max(pstar,smallp);
    Real pstnm1 = pstar;

    Real wlsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pl)+pstar)*rl;
    Real wrsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pr)+pstar)*rr;

    wl = std::sqrt(wlsq);
    wr = std::sqrt(wrsq);
    Real ustarp = ul - (pstar-pl)/wl;
    Real ustarm = ur + (pstar-pr)/wr;

    pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = max(pstar,smallp);

    Real ustar;
    for (int iter = 0; iter < 3; ++iter)
    {
        wlsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pl)+pstar)*rl;
        wrsq = (Real(0.5)*(gamma-Real(1.))*(pstar+pr)+pstar)*rr;

        wl = Real(1.)/std::sqrt(wlsq);
        wr = Real(1.)/std::sqrt(wrsq);

        Real ustnm1 = ustarm;
        Real ustnp1 = ustarp;

        ustarm = ur - (pr - pstar)*wr;
        ustarp = ul + (pl - pstar)*wl;

        Real dpditer = Math::abs(pstnm1-pstar);
        Real zp = Math::abs(ustarp-ustnp1);
        if (zp-weakwv*cleft < Real(0.0) ) {
            zp = dpditer*wl;
        }
        Real zm = Math::abs(ustarm-ustnm1);
        if (zm-weakwv*cright < Real(0.0) ) {
            zm = dpditer*wr;
        }

        Real zz = zp+zm;
        Real denom = dpditer/ max(zz,ccsmall);
        pstnm1 = pstar;
        pstar = pstar - denom*(ustarm-ustarp);
        pstar = max(pstar,smallp);
        ustar = Real(0.5)*(ustarm+ustarp);
    }

    Real ro, uo, po, sgnm, utrans1, utrans2;
    if (ustar > Real(0.)) {
        ro = rl;
        uo = ul;
        po = pl;
        sgnm = Real(1.);
        utrans1 = ut1l;
        utrans2 = ut2l;
    } else if (ustar < Real(0.)) {
        ro = rr;
        uo = ur;
        po = pr;
        sgnm = Real(-1.);
        utrans1 = ut1r;
        utrans2 = ut2r;
    } else {
        uo = Real(0.5)*(ur+ul);
        po = Real(0.5)*(pr+pl);
        ro = Real(2.)*(rl*rr)/(rl+rr);
        sgnm = Real(1.);
        utrans1 = Real(0.5)*(ut1l+ut1r);
        utrans2 = Real(0.5)*(ut2l+ut2r);
    }
    Real wosq = (Real(0.5)*(gamma-Real(1.))*(pstar+po)+pstar)*ro;
    Real co = std::sqrt(gamma * po / ro);
    Real wo = std::sqrt(wosq);
    Real dpjmp = pstar-po;
    Real rstar = ro/(Real(1.)-ro*dpjmp/wosq);
    Real cstar = std::sqrt(gamma * pstar / rstar);
    Real spout = co-sgnm*uo;
    Real spin = cstar - sgnm*uo;
    if(pstar >= po) {
        spin = wo/ro-sgnm*uo;
        spout = spin;
    }
    Real ss = max(spout-spin, spout+spin);
    Real frac = Real(0.5)*(Real(1.)+(spin+spout)/max(ss,ccsmall));

    Real rgdnv, ugdnv, pgdnv;
    if (spout < Real(0.)) {
        rgdnv = ro;
        ugdnv = uo;
        pgdnv = po;
    } else if(spin >= Real(0.)) {
        rgdnv = rstar;
        ugdnv = ustar;
        pgdnv = pstar;
    } else {
        rgdnv = frac*rstar + (Real(1.) - frac)* ro;
        ugdnv = frac*ustar + (Real(1.) - frac)* uo;
        pgdnv = frac*pstar + (Real(1.) - frac)* po;
    }

    flxrho = rgdnv*ugdnv;
    flxu = rgdnv*ugdnv*ugdnv+pgdnv;
    flxut = rgdnv*ugdnv*utrans1;
    flxutt = rgdnv*ugdnv*utrans2;
    flxe = ugdnv*(Real(0.5)*rgdnv*(ugdnv*ugdnv+utrans1*utrans1+utrans2*utrans2) + pgdnv/(gamma -Real(1.)) + pgdnv);
}

AMREX_GPU_DEVICE
inline
void
cns_riemann_x (int i, int j, int k,
               Array4<Real> const& fx,
               Array4<Real const> const& dq,
               Array4<Real const> const& q,
               Parm const& parm) noexcept
{
    

    Real cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i-1,j,k,QT)) + 1.e-40;
    Real rl = q(i-1,j,k,QRHO) + Real(0.5) * ( (dq(i-1,j,k,0)+dq(i-1,j,k,2))/cspeed + dq(i-1,j,k,1));
    rl = max(rl, parm.smallr);
    Real ul = q(i-1,j,k,QU) + Real(0.5) * ( (dq(i-1,j,k,2)-dq(i-1,j,k,0))/q(i-1,j,k,QRHO));
    Real pl = q(i-1,j,k,QPRES) + Real(0.5) *  (dq(i-1,j,k,0)+dq(i-1,j,k,2))*cspeed;
    pl = max(pl, parm.smallp);
    Real ut1l = q(i-1,j,k,QV) + Real(0.5) * dq(i-1,j,k,3);
    Real ut2l = q(i-1,j,k,QW) + Real(0.5) * dq(i-1,j,k,4);

    cspeed =  sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k,QT)) + 1.e-40;
    Real rr = q(i,j,k,QRHO) - Real(0.5) * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
    rr = max(rr, parm.smallr);
    Real ur = q(i,j,k,QU) - Real(0.5) * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
    Real pr = q(i,j,k,QPRES) - Real(0.5) * (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
    pr = max(pr, parm.smallp);
    Real ut1r = q(i,j,k,QV) - Real(0.5) * dq(i,j,k,3);
    Real ut2r = q(i,j,k,QW) - Real(0.5) * dq(i,j,k,4);

    riemann(parm.eos_gamma, parm.smallp, parm.smallr,
            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
            fx(i,j,k,URHO), fx(i,j,k,UMX), fx(i,j,k,UMY), fx(i,j,k,UMZ), fx(i,j,k,UET));
}

AMREX_GPU_DEVICE inline
void cns_riemann_y (int i, int j, int k,
               Array4<Real> const& fy,
               Array4<Real const> const& dq,
               Array4<Real const> const& q,
               Parm const& parm) noexcept
{
    

    Real cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i,j-1,k,QT)) + 1.e-40;
    Real rl = q(i,j-1,k,QRHO) + Real(0.5) * ( (dq(i,j-1,k,0)+dq(i,j-1,k,2))/cspeed + dq(i,j-1,k,1));
    rl = max(rl, parm.smallr);
    Real ul = q(i,j-1,k,QV) + Real(0.5) * ( (dq(i,j-1,k,2)-dq(i,j-1,k,0))/q(i,j-1,k,QRHO));
    Real pl = q(i,j-1,k,QPRES) + Real(0.5) *  (dq(i,j-1,k,0)+dq(i,j-1,k,2))*cspeed;
    pl = max(pl, parm.smallp);
    Real ut1l = q(i,j-1,k,QU) + Real(0.5) * dq(i,j-1,k,3);
    Real ut2l = q(i,j-1,k,QW) + Real(0.5) * dq(i,j-1,k,4);

    cspeed =  sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k,QT)) + 1.e-40;
    Real rr = q(i,j,k,QRHO) - Real(0.5) * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
    rr = max(rr, parm.smallr);
    Real ur = q(i,j,k,QV) - Real(0.5) * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
    Real pr = q(i,j,k,QPRES) - Real(0.5) * (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
    pr = max(pr, parm.smallp);
    Real ut1r = q(i,j,k,QU) - Real(0.5) * dq(i,j,k,3);
    Real ut2r = q(i,j,k,QW) - Real(0.5) * dq(i,j,k,4);

    riemann(parm.eos_gamma, parm.smallp, parm.smallr,
            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
            fy(i,j,k,URHO), fy(i,j,k,UMY), fy(i,j,k,UMX), fy(i,j,k,UMZ), fy(i,j,k,UET));
}

AMREX_GPU_DEVICE inline 
void cns_riemann_z (int i, int j, int k,
               Array4<Real> const& fz,
               Array4<Real const> const& dq,
               Array4<Real const> const& q,
               Parm const& parm) noexcept
{
    Real cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k+1,QT)) + 1.e-40;
    Real rl = q(i,j,k-1,QRHO) + Real(0.5) * ( (dq(i,j,k-1,0)+dq(i,j,k-1,2))/cspeed + dq(i,j,k-1,1));
    rl = max(rl, parm.smallr);
    Real ul = q(i,j,k-1,QW) + Real(0.5) * ( (dq(i,j,k-1,2)-dq(i,j,k-1,0))/q(i,j,k-1,QRHO));
    Real pl = q(i,j,k-1,QPRES) + Real(0.5) *  (dq(i,j,k-1,0)+dq(i,j,k-1,2))*cspeed;
    pl = max(pl, parm.smallp);
    Real ut1l = q(i,j,k-1,QU) + Real(0.5) * dq(i,j,k-1,3);
    Real ut2l = q(i,j,k-1,QV) + Real(0.5) * dq(i,j,k-1,4);

    cspeed = sqrt(parm.eos_gamma*parm.Rspec*q(i,j,k,QT)) + 1.e-40;
    Real rr = q(i,j,k,QRHO) - Real(0.5) * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
    rr = max(rr, parm.smallr);
    Real ur = q(i,j,k,QW) - Real(0.5) * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
    Real pr = q(i,j,k,QPRES) - Real(0.5) *  (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
    pr = max(pr, parm.smallp);
    Real ut1r = q(i,j,k,QU) - Real(0.5) * dq(i,j,k,3);
    Real ut2r = q(i,j,k,QV) - Real(0.5) * dq(i,j,k,4);

    riemann(parm.eos_gamma, parm.smallp, parm.smallr,
            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
            fz(i,j,k,URHO), fz(i,j,k,UMZ), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,UET));
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
