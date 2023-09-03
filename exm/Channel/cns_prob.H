#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "CNS_parm.H"
#include "cns_prob_parm.H"
#include "CNS_index_macros.H"

using namespace amrex;

AMREX_GPU_DEVICE inline
void prob_initdata (int i, int j, int k, Array4<Real> const& state, GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm) {
  const Real* prob_lo = geomdata.ProbLo();
  const Real* prob_hi = geomdata.ProbHi();
  const Real* dx      = geomdata.CellSize();

  Real pi = 3.14159265358979323846_rt;
  Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
  Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
  Real z = prob_lo[2] + (k+Real(0.5))*dx[2];

  Real lx = prob_hi[0] - prob_lo[0];
  Real ly = prob_hi[1] - prob_lo[1];
  Real lz = prob_hi[2] - prob_lo[2];

  if (y > prob_parm.h) {y = 2*prob_parm.h-y;}

  // Laminar velocity and constant temperature
  // Real ux = 1.5*prob_parm.ub*(1 - (1/std::pow(prob_parm.h,2))*std::pow((y-prob_parm.h),2));
  // Real ux = 0.0_rt;

  // Real T = prob_parm.Tw+0.5*prob_parm.Tw*(1 - (1/std::pow(prob_parm.h,2))*std::pow((y-prob_parm.h),2));
  // Real T = prob_parm.Tw;

  // Turbulent profiles
  Real T = 677.04110513 + 255.46817*pow(y,0.5) - 176.38967315*exp(-3081.95957181*y)*1.025;
  Real ux = 581.54524091 + 2687.84321585*pow(y,0.5) - 593.73493364* exp(-2008.87100276*y);

  int nx = 1;
  int ny = 2;
  int nz = 3;

  // srand(10);
  // Print() << (Real)rand()/ RAND_MAX << std::endl;
  // amrex::RandomEngine re;
// Real RandomNormal (Real mean, Real stddev, RandomEngine const& random_engine)
  // Real rand_num = 1.0;//amrex::RandomNormal(0.0, 0.0,re);
#if AMREX_USE_GPU
  Real rand_num = 0.5;
#else
  Real rand_num = (Real)rand()/ RAND_MAX;
#endif

  Real A       = 100*2.5;
  Real u_prime = rand_num*A*cos(2*pi*x*nx/lx)*sin(pi*y*ny/ly)*cos(2*pi*z*nz/lz);
  Real v_prime = rand_num*-A*sin(2*pi*x*nx/lx)*cos(pi*y*ny/ly)*sin(2*pi*z*nz/lz);
  Real w_prime = rand_num*-A*sin(2*pi*x*nx/lx)*cos(pi*y*ny/ly)*sin(2*pi*z*nz/lz);

  Real rho =  prob_parm.Pw/(parm.Rspec*T);
  state(i,j,k,URHO ) = rho;
  state(i,j,k,UMX  ) = rho*(ux+u_prime);
  state(i,j,k,UMY  ) = rho*v_prime;
  state(i,j,k,UMZ  ) = rho*w_prime;
  state(i,j,k,UET  ) = rho*parm.cv*T + Real(0.5)*rho*(ux*ux + u_prime*u_prime + v_prime*v_prime + w_prime*w_prime);
}

AMREX_GPU_DEVICE inline 
void user_tagging(int i, int j, int k, auto& tagfab, const auto &sdatafab, const auto& geomdata, const ProbParm& prob_parm , int level) {

    int jmax = geomdata.Domain().bigEnd(1);
    int jmin = geomdata.Domain().smallEnd(1);

    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();
    Real y = (j+Real(0.5))*dx[1];
    Real factor = 0.5;
    tagfab(i,j,k) = y < prob_parm.h*factor || y > prob_hi[1] - prob_parm.h*factor;
  }


AMREX_GPU_DEVICE inline
void user_source(int i, int j, int k, const auto& state, const auto& rhs, const ProbParm& lprobparm,Parm const& parms, auto const dx) {

  // Print() << lprobparm.fx << std::endl;
  rhs(i,j,k,UMX) +=  state(i,j,k,URHO)*lprobparm.fx;
  rhs(i,j,k,UET) +=  state(i,j,k,UMX )*lprobparm.fx;

}


/**
 * \brief Fill external boundary conditions for ghost cells.
 *
 * @param x         ghost cell cooridinates.
 * @param dr        wall-ghost/wall-first internal distance ratio 
 * @param s_int     flow state inside of the domain.
 * @param s_ext     flow state to be filled.
 * @param idir      direction (0: x, 1: y, 2: z).
 * @param sgn       high or low boundary (1: low, -1: high).
 * @param time      time.
 * @param geomdata  domain geometry data.
 * @param prob_parm ProbParm data as defined in prob_parm.H and initialised in
 * amrex_probinit.
 * @sa CnsFillExtDir
 * @sa CnsFillExtDir::operator()
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
bcnormal(const amrex::Real x[AMREX_SPACEDIM], amrex::Real dratio, const amrex::Real s_int[NCONS],
         const amrex::Real s_refl[NCONS], amrex::Real s_ext[NCONS],
         const int idir, const int sgn, const amrex::Real time,
         amrex::GeometryData const& /*geomdata*/,  Parm const& parm, ProbParm const& prob_parm)
{
  if (idir == 1) { // ylo or yhi

    Real q_ext[NPRIM] = {0.0};
    // no-slip
    q_ext[QU]    = -s_int[UMX]/s_int[URHO];
    q_ext[QV]    = -s_int[UMY]/s_int[URHO];
    q_ext[QW]    = -s_int[UMZ]/s_int[URHO];

    // dp/dn = 0
    amrex::Real eint_int = (s_int[UET] - 0.5*(s_int[UMX]*s_int[UMX] + s_int[UMY]*s_int[UMY] + s_int[UMZ]*s_int[UMZ])/s_int[URHO])/s_int[URHO];
    amrex::Real p_int = (parm.eos_gamma - 1.0)*s_int[URHO]*eint_int;
    q_ext[QPRES] = p_int;
    // T=Twall
    amrex::Real T_int = p_int/(parm.Rspec*s_int[URHO]); 
    q_ext[QT]    = max(prob_parm.Tw  +  dratio*(prob_parm.Tw - T_int),50.0);
    // rho = eos(P,T)
    q_ext[QRHO]  = q_ext[QPRES]/(parm.Rspec*q_ext[QT]);

    // convert prims to cons
    s_ext[URHO] = q_ext[QRHO];
    s_ext[UMX] = q_ext[QRHO]*q_ext[QU];
    s_ext[UMY] = q_ext[QRHO]*q_ext[QV];
    s_ext[UMZ] = q_ext[QRHO]*q_ext[QW];
    amrex::Real ekin_ext = 0.5*(q_ext[QU]*q_ext[QU] + q_ext[QV]*q_ext[QV] + q_ext[QW]*q_ext[QW]); 
    amrex::Real eint_ext = q_ext[QPRES]/(q_ext[QRHO]*(parm.eos_gamma - 1.0));
    s_ext[UET] = q_ext[QRHO]*(eint_ext + ekin_ext);
  }
}

#endif
