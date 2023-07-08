#ifndef CNS_PROB_PARM_H
#define CNS_PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex;
struct ProbParm
{
  Real Tw = 0.0; // not used but needed because of how BCs are implemented currently --> TO IMPROVE

  const Real pi = 3.1415926535897932384626433_rt;
  Real Lambda = 1.0_rt; // length of the domain, assuming it is a cube
  const Real ampl_p = 1.0_rt;
  const Real ampl_t = 1.0_rt;
  const Real ampl_u = 1.0_rt;
  const Real ampl_v = 1.0_rt;
  const Real ampl_w = 1.0_rt;
  const Real n_p = 1.0_rt;
  const Real n_t = 1.0_rt;
  const Real n_u = 1.0_rt;
  const Real n_v = 1.0_rt;
  const Real n_w = 1.0_rt;

  const Real alpha_p = 2.0_rt*n_p*pi/Lambda;
  const Real alpha_t = 2.0_rt*n_t*pi/Lambda;
  const Real alpha_u = 2.0_rt*n_u*pi/Lambda;
  const Real alpha_v = 2.0_rt*n_v*pi/Lambda;
  const Real alpha_w = 2.0_rt*n_w*pi/Lambda;

  Real fu (Real x, Real y, Real z) const {
    return ampl_u * cos(alpha_u*x)*sin(alpha_u*y)*sin(alpha_u*z);
  }

  Real fv (Real x, Real y, Real z) const {
    return ampl_v * cos(alpha_v*x)*sin(alpha_v*y)*sin(alpha_v*z);
  }

  Real fw (Real x, Real y, Real z) const {
    return ampl_w * cos(alpha_w*x)*sin(alpha_w*y)*sin(alpha_w*z);
  }

  Real ft (Real x, Real y, Real z) const {
    return 4.0_rt + ampl_t * cos(alpha_t*x)*sin(alpha_t*y)*cos(alpha_t*z);
  }

  Real fp (Real x, Real y, Real z) const {
    return 4.0_rt + ampl_p * cos(alpha_p*x)*sin(alpha_p*y)*cos(alpha_p*z);
  }

  Real dudx (Real x, Real y, Real z) const {
    return -ampl_u * alpha_u * sin(alpha_u*x) * sin(alpha_u*y) * sin(alpha_u*z) ;}

  Real dudy (Real x, Real y, Real z) const {
    return ampl_u * alpha_u * cos(alpha_u*x) * cos(alpha_u*y) * sin(alpha_u*z) ;}
  
  Real dudz (Real x, Real y, Real z) const {
    return ampl_u * alpha_u * cos(alpha_u*x) * sin(alpha_u*y) * cos(alpha_u*z) ;}

  Real dvdx (Real x, Real y, Real z) const {
    return - ampl_v * alpha_v * sin(alpha_v*x) * sin(alpha_v*y) * sin(alpha_v*z) ;}

  Real dvdy (Real x, Real y, Real z) const {
    return  ampl_v * alpha_v * cos(alpha_v*x) * cos(alpha_v*y) * sin(alpha_v*z) ;}

  Real dvdz (Real x, Real y, Real z) const {
    return  ampl_v * alpha_v * cos(alpha_v*x) * sin(alpha_v*y) * cos(alpha_v*z) ;}

  Real dwdx (Real x, Real y, Real z) const {
    return - ampl_w * alpha_w * sin(alpha_w*x) * sin(alpha_w*y) * sin(alpha_w*z) ;}

  Real dwdy (Real x, Real y, Real z) const {
    return  ampl_w * alpha_w * cos(alpha_w*x) * cos(alpha_w*y) * sin(alpha_w*z) ;}

  Real dwdz (Real x, Real y, Real z) const {
    return  ampl_w * alpha_w * cos(alpha_w*x) * sin(alpha_w*y) * cos(alpha_w*z) ;}

  Real dpdx (Real x, Real y, Real z) const {
    return - ampl_p * alpha_p * sin(alpha_p*x) * sin(alpha_p*y) * cos(alpha_p*z) ;}

  Real dpdy (Real x, Real y, Real z) const {
    return ampl_p * alpha_p * cos(alpha_p*x) * cos(alpha_p*y) * cos(alpha_p*z) ;}

  Real dpdz (Real x, Real y, Real z) const {
    return - ampl_p * alpha_p * cos(alpha_p*x) * sin(alpha_p*y) * sin(alpha_p*z);}

  Real dTdx (Real x, Real y, Real z) const {
    return - ampl_t * alpha_t * sin(alpha_t*x) * sin(alpha_t*y) * cos(alpha_t*z) ;}

  Real dTdy (Real x, Real y, Real z) const {
    return ampl_t * alpha_t * cos(alpha_t*x) * cos(alpha_t*y) * cos(alpha_t*z) ;}

  Real dTdz (Real x, Real y, Real z) const {
    return - ampl_t * alpha_t * cos(alpha_t*x) * sin(alpha_t*y) * sin(alpha_t*z);}

};

#endif
