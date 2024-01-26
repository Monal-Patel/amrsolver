#ifndef CLOSURES_H_
#define CLOSURES_H_

#include <AMReX_MFIter.H>

using namespace amrex;

// Independent (solved) variables
#define URHO 0
#define UMX 1
#define UMY 2
#define UMZ 3
#define UET 4
#define NCONS 5

// Dependent (derived) variables
#define QRHO 0
#define QU 1
#define QV 2
#define QW 3
#define QT 4
#define QPRES 5
#define NPRIM 6
#define NGHOST 3  // TODO: make this an automatic parameter?

////////////////////////////////THERMODYNAMICS/////////////////////////////////
class calorifically_perfect_gas_nasg_liquid_t
{
private:
  /* data */
  // gamma_a =;
  // gamma_l =;

public:
// state

Real inline energy (Real p, Real rho) {
  Real eint=0;
return eint;
}

// pressure (eint,rho)

};


class calorifically_perfect_gas_t {
 public:
  Real gamma = 1.40;   // ratio of specific heats
  Real mw = 28.96e-3;  // mean molecular weight air kg/mol

  Real Ru = Real(8.31451);
  Real cv = Ru / (mw * (gamma - Real(1.0)));
  Real cp = gamma * Ru / (mw * (gamma - Real(1.0)));
  Real Rspec = Ru / mw;

  Real sos(Real& T){return std::sqrt(this->gamma * this->Rspec * T);}

  // void prims2cons(i,j,k,){};

  void prims2char(){};


  // void prims2flux(int& i, int& j, int& k, const GpuArray<Real,NPRIM>& prims, GpuArray<Real,NCONS>& fluxes, const GpuArray<int, 3>& vdir) {
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE
  void prims2fluxes(int& i, int& j, int& k, const Array4<Real>& prims, Array4<Real>& fluxes, const GpuArray<int, 3>& vdir) {

    Real rho = prims(i,j,k,QRHO);
    Real ux  = prims(i,j,k,QU);
    Real uy  = prims(i,j,k,QV);
    Real uz  = prims(i,j,k,QW);
    Real P   = prims(i,j,k,QPRES);
    Real udir = ux*vdir[0] + uy*vdir[1] + uz*vdir[2];

    Real ekin  = Real(0.5) *(ux*ux + uy*uy + uz*uz);
    Real rhoet = rho*(this->cp*prims(i,j,k,QT) + ekin) ;

    fluxes(i,j,k,URHO) = rho*udir;
    fluxes(i,j,k,UMX)  = rho*ux*udir + P*vdir[0];
    fluxes(i,j,k,UMY)  = rho*uy*udir + P*vdir[1];
    fluxes(i,j,k,UMZ)  = rho*uz*udir + P*vdir[2];
    fluxes(i,j,k,UET)  = (rhoet + P) * udir;
  };

  // can move this to closures derived tyoe (closures_dt)
  // prims to cons
  // - We want to call it from thermodynamics class
  // - cls is stored on cpu and gpu
  // TODO: remove ParallelFor from here. Keep closures local
  void inline cons2prims(const MFIter& mfi, const Array4<Real>& cons,
                         const Array4<Real>& prims) const {
    const Box& bxg = mfi.growntilebox(NGHOST);

    amrex::ParallelFor(bxg, [=, *this] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real rho = cons(i, j, k, URHO);
      // Print() << "cons2prim"<< i << j << k << rho << std::endl;
      rho = max(1e-40, rho);
      Real rhoinv = Real(1.0) / rho;
      Real ux = cons(i, j, k, UMX) * rhoinv;
      Real uy = cons(i, j, k, UMY) * rhoinv;
      Real uz = cons(i, j, k, UMZ) * rhoinv;
      Real rhoke = Real(0.5) * rho * (ux * ux + uy * uy + uz * uz);
      Real rhoei = (cons(i, j, k, UET) - rhoke);
      Real p = (this->gamma - Real(1.0)) * rhoei;

      prims(i, j, k, QRHO) = rho;
      prims(i, j, k, QU) = ux;
      prims(i, j, k, QV) = uy;
      prims(i, j, k, QW) = uz;
      prims(i, j, k, QPRES) = p;
      prims(i, j, k, QT) = p / (rho * (this->Rspec));
    });
  }
};


////////////////////////////////TRANSPORT/////////////////////////////////
////// Viscosity
class visc_const_t {
 private:
 public:
  Real visc_ref = 1.458e-6;  // Viscosity reference value

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real visc(const Real& T) const {
    return visc_ref;
  }
};

class visc_suth_t {
 private:
  // Sutherland's fit from Computational Fluid Mechanics and Heat Transfer
  Real visc_ref = 1.458e-6;
  Real Tvisc_ref = 110.4;

 public:
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real visc(const Real& T) const {
    return visc_ref * T * sqrt(T) / (Tvisc_ref + T);
  }
};

////// Conductivity
class cond_const_t {
 private:
 public:
  Real cond_ref = 1.458e-6;

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real cond(Real& T) const {
    return cond_ref;
  }

#ifdef AMREX_USE_GPU
  AMREX_FORCE_INLINE Real cond_cpu(Real& T) const { return cond_ref; }
#endif
};

// class cond_const_pr_t private: ThermodynamicsBase {
//   private:
//   public:
//     Real visc_ref = 1.458e-6; // Viscosity reference value

//     AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real cond(Real& T) const {
//       // cp*;
//     return visc_ref;}
// };

class cond_suth_t {
 private:
  // Sutherland's fit from Computational Fluid Mechanics and Heat Transfer
  Real cond_ref = 2.495e-3;
  Real Tcond_ref = 194.0;

 public:
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real cond(Real& T) const {
    return cond_ref * T * sqrt(T) / (Tcond_ref + T);
  }

#ifdef AMREX_USE_GPU
  AMREX_FORCE_INLINE Real cond_cpu(Real& T) const {
    return cond_ref * T * sqrt(T) / (Tcond_ref + T);
  }
#endif
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////CLOSURES/////////////////////////////////
template <typename Visc, typename Cond, typename Thermo, typename... others>
class closures_dt : public Cond, public Visc, public Thermo, public others... {
 private:
 public:
  Real Cshock = 1.0;
  Real Cdamp = 0.01;
};
#endif