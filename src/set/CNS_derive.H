#ifndef CNS_DERIVE_H_
#define CNS_DERIVE_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

using namespace amrex;

void derpres (const Box& bx, FArrayBox& derfab, 
                  int dcomp, int ncomp, const FArrayBox& datafab, const Geometry& geomdata, Real time, const int* bcrec, int level);

void dertemp (const Box& bx, FArrayBox& derfab, int dcomp, int ncomp,
                  const FArrayBox& datafab, const Geometry& geomdata,
                  Real time, const int* bcrec, int level);

void dervel (const Box& bx, FArrayBox& derfab, 
              int dcomp, int ncomp, const FArrayBox& datafab, const Geometry& geomdata, Real time, const int* bcrec, int level);

#endif
