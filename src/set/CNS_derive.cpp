#include "CNS_derive.H"
#include "CNS.H"
#include <CNS_index_macros.H>

using namespace amrex;

void derpres (const Box& bx, FArrayBox& derfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datafab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat  = datafab.array();
    auto pfab       = derfab.array(dcomp);
    Parm const* parm = CNS::d_parm;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real rhoinv = 1.0_rt / dat(i, j, k, URHO);
        Real mx = dat(i, j, k, UMX);
        Real my = dat(i, j, k, UMY);
        Real mz = dat(i, j, k, UMZ);
        Real rhoeint = dat(i, j, k, UET) - Real(0.5)*rhoinv*(mx*mx + my*my + mz*mz); 
        pfab(i,j,k) = (parm->eos_gamma-Real(1.0))*rhoeint;
    });
}

void dertemp (const Box& bx, FArrayBox& derfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datafab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat  = datafab.array();
    auto Tfab       = derfab.array(dcomp);
    Parm const* parm = CNS::d_parm;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real rhoinv = 1.0_rt / dat(i, j, k, URHO);
        Real mx = dat(i, j, k, UMX);
        Real my = dat(i, j, k, UMY);
        Real mz = dat(i, j, k, UMZ);
        Real rhoeint = dat(i, j, k, UET) - Real(0.5)*rhoinv*(mx*mx + my*my + mz*mz); 
        Tfab(i,j,k) = (rhoeint*rhoinv)/parm->cv;
    });
}

void dervel(const Box& bx, FArrayBox& derfab, int dcomp, int /*ncomp*/,     
                const FArrayBox& datfab, const Geometry& /*geomdata*/, Real /*time*/, const int* /*bcrec*/, const int /*level*/)
{
    auto const dat = datfab.const_array();
    auto vel = derfab.array();

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      vel(i, j, k, dcomp) = dat(i,j,k,1)/dat(i,j,k,0);
    });
}