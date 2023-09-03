#ifndef CNS_PARM_H_
#define CNS_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct Parm
{
    amrex::Real eos_gamma = 1.40; // ratio of specific heats
    amrex::Real eos_mu = 28.96e-3;// mean molecular weight air kg/mol

    //Sutherland's fit from Computational Fluid Mechanics and Heat Transfer 
    amrex::Real visc_ref  = 1.458e-6; // Viscosity reference value
    amrex::Real Tvisc_ref = 110.4;    // Viscosity reference temperature
    amrex::Real cond_ref  = 2.495e-3; // Conductivity reference value
    amrex::Real Tcond_ref = 194.0;    // Conductivity reference temperature

    amrex::Real smallr = 1.e-19;
    amrex::Real smallp = 1.e-10;

    amrex::Real Ru = amrex::Real(8.31451);
    amrex::Real cv = Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    amrex::Real cp = eos_gamma * Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    amrex::Real Rspec = Ru/eos_mu;

    amrex::Real Pr = 0.70;

    amrex::Real Cshock = 0.0;
    amrex::Real Cdamp = 0.0;

    public: 
    // AMREX_GPU_DEVICE inline const amrex::Real visc(amrex::Real& T,mu) { 
    AMREX_GPU_DEVICE inline amrex::Real ViscositySutherland(amrex::Real& T) const { return visc_ref*T*sqrt(T)/(Tvisc_ref + T);}

    AMREX_GPU_DEVICE inline amrex::Real ConductivitySutherland(amrex::Real& T) const { return cond_ref*T*sqrt(T)/(Tcond_ref + T);}

    AMREX_GPU_DEVICE inline amrex::Real ConductivityConstPr(amrex::Real& T) const { return cp*ViscositySutherland(T)/Pr;}

    amrex::Real visc_CPU(amrex::Real T) { return visc_ref*T*sqrt(T)/(Tvisc_ref + T);}


    // template<typename T> AMREX_GPU_DEVICE 
    // constexpr auto sqrt(amrex::Real& T) noexcept {
    //     if (std::is_constant_evaluated())
    //         return ConductivitySutherland(T);
    //     else
    //         return ConductivityConstPr(T);
    // }

    // amrex::Real (*Parm::Viscosity) (amrex::Real) = NULL; 

    // inline amrex::Real cond_CPU(amrex::Real T) const { return cond_ref*T*sqrt(T)/(Tcond_ref + T);}

    // Declaring
    // amrex::Real (*Conductivity) (amrex::Real& T) = NULL; 

    // typedef amrex::Real (*TransportPropertiesPointer)(amrex::Real T);

    // TransportPropertiesPointer Viscosity =&visc_CPU;

    // int multiply(int a, int b) { return a * b; };


    // Referencing
    // EulerFlux=function_name;

    // Dereferencing
    // data_type x=*FuncPtr;

 

    // void SetPointers() {

    //   Viscosity = &Parm::visc_CPU;
      // auto fn_address = &Parm::visc_CPU; // get the address of the function
      // printf("Function address: %p\n", fn_address);
      // exit(0);

      // using TransportPointer = amrex::Real (*)(amrex::Real T);
      // TransportPointer Viscosity;
      // *(this->Viscosity) = visc_CPU;
      // Conductivity = &Parm::ConductivitySutherland;

      // std::cout << "The value of the product is: " << resultOne << std::endl;
      // exit(0);
    // }

};

#endif
