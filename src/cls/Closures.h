#include <AMReX.H>

using namespace amrex;


////////////////////////////////THERMODYNAMICS/////////////////////////////////
class EosBase
{
private:
  EosBase(/* args */);
  ~EosBase();

  /* data */ 
public:

  Real virtual pressure(){
    return 0.0;
  }

  Real virtual density() {
    return 0.0;
  }
  Real virtual energy() {
    return 0.0;
  }

};

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

class FluidState
{
private:
  // cons
  // prims

public:
  // FluidState(/* args */);
  // ~FluidState();

  //cons2prims
  //prims2cons
};

class calorifically_perfect_gas_t 
{
  public:
  Real gamma = 1.40; // ratio of specific heats
  Real mw = 28.96e-3;// mean molecular weight air kg/mol

  Real Ru = Real(8.31451);
  Real cv = Ru / (mw * (gamma-Real(1.0)));
  Real cp = gamma * Ru / (mw * (gamma-Real(1.0)));
  Real Rspec = Ru/mw;
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
    Real visc_ref = 1.458e-6; // Viscosity reference value

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real visc(Real& T) const {
    return visc_ref;}
};

class visc_suth_t {
  private:
    //Sutherland's fit from Computational Fluid Mechanics and Heat Transfer 
    Real visc_ref  = 1.458e-6;
    Real Tvisc_ref = 110.4;
  public:
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real visc(Real& T) const {
    return visc_ref*T*sqrt(T)/(Tvisc_ref + T);}
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
    return cond_ref;}

#ifdef AMREX_USE_GPU
    AMREX_FORCE_INLINE Real cond_cpu(Real& T) const {
    return cond_ref;}
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
    //Sutherland's fit from Computational Fluid Mechanics and Heat Transfer 
    Real cond_ref  = 2.495e-3;
    Real Tcond_ref = 194.0;
  public:
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE Real cond(Real& T) const { return cond_ref*T*sqrt(T)/(Tcond_ref + T);}

#ifdef AMREX_USE_GPU
    AMREX_FORCE_INLINE Real cond_cpu(Real& T) const { return cond_ref*T*sqrt(T)/(Tcond_ref + T);}
#endif
};


////////////////////////////////////////////////////////////////////////////////

////////////////////////////////CLOSURES/////////////////////////////////
template <typename Visc, typename Cond, typename Thermo, typename ProbParms >
class closures_derived_base_t : public Cond, public Visc, public Thermo, public ProbParms
{
  private:
  public:
    // IdealPerfectGas(/* args */);
    // ~IdealPerfectGas();
  Real Cshock = 1.0;
  Real Cdamp  = 0.01;

};


// Class MultispeciesIdealGas : public BaseEOS, public ThermodynamicsBase, public TransportBase
// {
//   private:
//     MultispeciesIdealGas(/* args */);
//     ~MultispeciesIdealGas();
//   public:
// };


// class Multiphase : public BaseEOS, public ThermodynamicsBase, public TransportBase
// {
//   private:
//     Multiphase(/* args */);
//     ~Multiphase();
//   public:


// };