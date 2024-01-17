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

//
#define NGHOST 3  // TODO: make this an automatic parameter?

////////////////////////////////THERMODYNAMICS/////////////////////////////////
// class EosBase {
//  private:
//   EosBase(/* args */);
//   ~EosBase();

//   /* data */
//  public:
//   Real virtual pressure() { return 0.0; }

//   Real virtual density() { return 0.0; }
//   Real virtual energy() { return 0.0; }
// };

// class stiffgas_t : public EosBase
// {
// private:
//   /* data */
//   gamma_a =;
//   gamma_l =;

// public:
// // state

// Real energy (Real p, Real rho) {

// return eint
// }

// // pressure (eint,rho)

// }

//
// class ThermodynamicsBase
// {
// private:
//   /* data */
// public:
//   ThermodynamicsBase(/* args */);
//   ~ThermodynamicsBase();
// };

class calorifically_perfect_gas_t {
 public:
  Real gamma = 1.40;   // ratio of specific heats
  Real mw = 28.96e-3;  // mean molecular weight air kg/mol

  Real Ru = Real(8.31451);
  Real cv = Ru / (mw * (gamma - Real(1.0)));
  Real cp = gamma * Ru / (mw * (gamma - Real(1.0)));
  Real Rspec = Ru / mw;

  // operates
  void prims2cons(){

  };

  void cons2prims(){

  };

  // can move this to closures derived tyoe (closures_dt)
  // prims to cons
  // - We want to call it from thermodynamics class
  // - cls is stored on cpu and gpu
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

// class TransportBase
// {
// private:
//   /* data */
// public:
//   TransportBase(/* args */);
//   ~TransportBase();

// };

////// Viscosity
// enum ViscOptionsEnum {visc_cons_e, visc_suth_e, visc_other_e};
// constexpr static ViscOptionsEnum ViscSelectUser = visc_suth_e;

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

// auto select_viscosity_law()
// {
//   if constexpr (ViscSelectUser == visc_cons_e) return visc_const_t();
//   else if constexpr (ViscSelectUser == visc_suth_e) return visc_suth_t();
//   else return 0 ;
// }

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

// Class MultispeciesIdealGas : public BaseEOS, public ThermodynamicsBase,
// public TransportBase
// {
//   private:
//     MultispeciesIdealGas(/* args */);
//     ~MultispeciesIdealGas();
//   public:
// };

// class Multiphase : public BaseEOS, public ThermodynamicsBase, public
// TransportBase
// {
//   private:
//     Multiphase(/* args */);
//     ~Multiphase();
//   public:
// };

#endif