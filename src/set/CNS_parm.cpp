
#include <CNS_parm.H>

void Parm::Initialize ()
{
    constexpr amrex::Real Ru = amrex::Real(8.31451);
    cv = Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    cp = eos_gamma * Ru / (eos_mu * (eos_gamma-amrex::Real(1.0)));
    Rspec = Ru/eos_mu;
}
