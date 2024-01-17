#ifndef MMS_H
#define MMS_H

#include <gtest/gtest.h>
#include <iostream>

#include <AMReX_BaseFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>
#include <Closures.h>
#include <RHS.h>

using namespace amrex;

///
/// \brief Test fixture class for MMS testing euler numerical schemes
///
class MMS : public testing::Test {
public:
  // arguments
  const int numcomps = 5;
  const int ndim = 3;
  BaseFab<Real> primsF, flx1F, flx2F, flx3F, rhsF;
  Box bx;

  template <class scheme_t, class cls_t>
  Real test_keep_scheme(const scheme_t scheme, const cls_t cls, const Array1D<Real, 0, 2>& dx) {

    // pepare arrays
    const Array4<Real> &prims = primsF.array();
    const Array4<Real> &flx1 = flx1F.array();
    const Array4<Real> &flx2 = flx2F.array();
    const Array4<Real> &flx3 = flx3F.array();
    const Array4<Real> &rhs = rhsF.array();

    // coefficients arrays (ugly, could be improved. Not sure how right now.)
    int idx;
    if (scheme.order_keep == 2) {
      idx = 0;
    } else if (scheme.order_keep == 4) {
      idx = 1;
    } else if (scheme.order_keep == 6) {
      idx = 2;
    } else {
      amrex::Abort("wrong scheme order");
    }
    Array1D<Real, 0, 2> coefs{scheme.coeffs(idx, 0), scheme.coeffs(idx, 1),
                              scheme.coeffs(idx, 2)};

    // fill prims
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real x = (i + 0.5_rt) * dx(0);
      Real y = (j + 0.5_rt) * dx(1);
      Real z = (k + 0.5_rt) * dx(2);

      Real P = fp(x, y, z);
      Real T = ft(x, y, z);
      prims(i, j, k, 0) = P / (cls.Rspec * T);
      prims(i, j, k, 1) = fu(x, y, z);
      prims(i, j, k, 2) = fv(x, y, z);
      prims(i, j, k, 3) = fw(x, y, z);
      prims(i, j, k, 4) = T;
      prims(i, j, k, 5) = P;
    });

    int i = 0;
    int j = 0;
    int k = 0;

    // numerical flux derivatives (rhs)
    scheme.flux_dir(i + 1, j, k, 0, coefs, prims, flx1, cls);
    scheme.flux_dir(i, j, k, 0, coefs, prims, flx1, cls);

    scheme.flux_dir(i, j + 1, k, 1, coefs, prims, flx2, cls);
    scheme.flux_dir(i, j, k, 1, coefs, prims, flx2, cls);

    scheme.flux_dir(i, j, k + 1, 2, coefs, prims, flx3, cls);
    scheme.flux_dir(i, j, k, 2, coefs, prims, flx3, cls);

    for (int nn = 0; nn < 5; nn++) {
      rhs(i, j, k, nn) = (flx1(i, j, k, nn) - flx1(i + 1, j, k, nn)) / dx(0) +
                         (flx2(i, j, k, nn) - flx2(i, j + 1, k, nn)) / dx(1) +
                         (flx3(i, j, k, nn) - flx3(i, j, k + 1, nn)) / dx(2);
      // std::cout << "nn, rhs = " << nn << " " << rhs(i, j, k, nn) << std::endl;
    }

    // error rhs
    const Real x = (i + 0.5_rt) * dx(0);
    const Real y = (j + 0.5_rt) * dx(1);
    const Real z = (k + 0.5_rt) * dx(2);
    Real rho = fp(x, y, z) / (cls.Rspec * ft(x, y, z));

    Real error = rhs(i, j, k, URHO) +
                 rho * (dudx(x, y, z) + dvdy(x, y, z) + dwdz(x, y, z));
    rhs(i, j, k, URHO) = error;

    // rho (duu/dx + duv/dy + duw/dz) assuming rho=constant
    error = rhs(i, j, k, UMX) +
            rho * (2 * fu(x, y, z) * dudx(x, y, z) +
                   fv(x, y, z) * dudy(x, y, z) + fu(x, y, z) * dvdy(x, y, z) +
                   fw(x, y, z) * dudz(x, y, z) + fu(x, y, z) * dwdz(x, y, z)) +
            dpdx(x, y, z);
    rhs(i, j, k, UMX) = error;

    error = rhs(i, j, k, UMY) +
            rho * (fu(x, y, z) * dvdx(x, y, z) + fv(x, y, z) * dudx(x, y, z) +
                   2 * fv(x, y, z) * dvdy(x, y, z) +
                   fw(x, y, z) * dvdz(x, y, z) + fv(x, y, z) * dwdz(x, y, z)) +
            dpdy(x, y, z);
    rhs(i, j, k, UMY) = error;

    error = rhs(i, j, k, UMZ) +
            rho * (fu(x, y, z) * dwdx(x, y, z) + fw(x, y, z) * dudx(x, y, z) +
                   fv(x, y, z) * dwdy(x, y, z) + fw(x, y, z) * dvdy(x, y, z) +
                   2 * fw(x, y, z) * dwdz(x, y, z)) +
            dpdz(x, y, z);
    rhs(i, j, k, UMZ) = error;

    // d (rho u ht) = d(rho u (et + p/rho)) = d(rho u et + Pu) = rho d(u et +
    // Pu/rho) = rho [ d(u et) + 1/rho d(Pu) ] = rho [ et d(u) + u d(et) +
    // (1/rho) ( u d(P) + P d(u) )]

    // et = e + 1/2 (u^2 + v^2 + w^2) = cv T + 1/2 (u^2 + v^2 + w^2)
    Real et = cls.cv * ft(x, y, z) +
              0.5_rt * (fu(x, y, z) * fu(x, y, z) + fv(x, y, z) * fv(x, y, z) +
                        fw(x, y, z) * fw(x, y, z));

    // x-direction: rho [ et d(u)/dx + u d(et)/dx + (1/rho) ( u d(P)dx + P
    // d(u)dx )] d(et)/dx = cv dT/dx + 2*0.5(udu/dx + vdv/dx + wdw/dx)
    Real det = cls.cv * dTdx(x, y, z) +
               (fu(x, y, z) * dudx(x, y, z) + fv(x, y, z) * dvdx(x, y, z) +
                fw(x, y, z) * dwdx(x, y, z));

    error = rho * (et * dudx(x, y, z) + fu(x, y, z) * det +
                   (1 / rho) * (fu(x, y, z) * dpdx(x, y, z) +
                                fp(x, y, z) * dudx(x, y, z)));

    // y-direction: rho [ et d(v) + v d(et) + (1/rho) ( v d(P) + P d(v) )]
    det = cls.cv * dTdy(x, y, z) +
          (fu(x, y, z) * dudy(x, y, z) + fv(x, y, z) * dvdy(x, y, z) +
           fw(x, y, z) * dwdy(x, y, z));
    error += rho * (et * dvdy(x, y, z) + fv(x, y, z) * det +
                    (1 / rho) * (fv(x, y, z) * dpdy(x, y, z) +
                                 fp(x, y, z) * dvdy(x, y, z)));

    // z-direction: rho [ et d(w) + w d(et) + (1/rho) ( w d(P) + P d(w) )]
    det = cls.cv * dTdz(x, y, z) +
          (fu(x, y, z) * dudz(x, y, z) + fv(x, y, z) * dvdz(x, y, z) +
           fw(x, y, z) * dwdz(x, y, z));

    error += rho * (et * dwdz(x, y, z) + fw(x, y, z) * det +
                    (1 / rho) * (fw(x, y, z) * dpdz(x, y, z) +
                                 fp(x, y, z) * dwdz(x, y, z)));

    error = error + rhs(i, j, k, UET);
    rhs(i, j, k, UET) = error;

    error = 0.0;
    for (int nn = 0; nn < 5; nn++) {
      // printf("error(%i)=%f \n",nn,rhs(i, j, k, nn));
      error = max(rhs(i, j, k, nn),error);
    }

    return error;
  }

protected:
  // You can remove any or all of the following functions if their bodies
  // would be empty.

  MMS() {
    // You can do set-up work for each test here.
    bx.setSmall(IntVect{-3, -3, -3});
    bx.setBig(IntVect{4, 4, 4});
    primsF.resize(bx, numcomps + 1);
    flx1F.resize(bx, numcomps);
    flx2F.resize(bx, numcomps);
    flx3F.resize(bx, numcomps);
    rhsF.resize(bx, numcomps);
  }

  ~MMS() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    std::cout << "setup up" << std::endl;
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
    std::cout << "tear down" << std::endl;
  }

  // void error(int i, int j, int k, const Array4<Real> &rhs, cls_t const &cls,
  //            const Array1D<Real, 0, 2> &dx) {

  // }

private:
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
  const Real alpha_p = 2.0_rt * n_p * pi / Lambda;
  const Real alpha_t = 2.0_rt * n_t * pi / Lambda;
  const Real alpha_u = 2.0_rt * n_u * pi / Lambda;
  const Real alpha_v = 2.0_rt * n_v * pi / Lambda;
  const Real alpha_w = 2.0_rt * n_w * pi / Lambda;

  Real fu(Real x, Real y, Real z) const {
    return ampl_u * cos(alpha_u * x) * sin(alpha_u * y) * sin(alpha_u * z);
  }

  Real fv(Real x, Real y, Real z) const {
    return ampl_v * cos(alpha_v * x) * sin(alpha_v * y) * sin(alpha_v * z);
  }

  Real fw(Real x, Real y, Real z) const {
    return ampl_w * cos(alpha_w * x) * sin(alpha_w * y) * sin(alpha_w * z);
  }

  Real ft(Real x, Real y, Real z) const {
    return 4.0_rt +
           ampl_t * cos(alpha_t * x) * sin(alpha_t * y) * cos(alpha_t * z);
  }

  Real fp(Real x, Real y, Real z) const {
    return 4.0_rt +
           ampl_p * cos(alpha_p * x) * sin(alpha_p * y) * cos(alpha_p * z);
  }

  Real dudx(Real x, Real y, Real z) const {
    return -ampl_u * alpha_u * sin(alpha_u * x) * sin(alpha_u * y) *
           sin(alpha_u * z);
  }

  Real dudy(Real x, Real y, Real z) const {
    return ampl_u * alpha_u * cos(alpha_u * x) * cos(alpha_u * y) *
           sin(alpha_u * z);
  }

  Real dudz(Real x, Real y, Real z) const {
    return ampl_u * alpha_u * cos(alpha_u * x) * sin(alpha_u * y) *
           cos(alpha_u * z);
  }

  Real dvdx(Real x, Real y, Real z) const {
    return -ampl_v * alpha_v * sin(alpha_v * x) * sin(alpha_v * y) *
           sin(alpha_v * z);
  }

  Real dvdy(Real x, Real y, Real z) const {
    return ampl_v * alpha_v * cos(alpha_v * x) * cos(alpha_v * y) *
           sin(alpha_v * z);
  }

  Real dvdz(Real x, Real y, Real z) const {
    return ampl_v * alpha_v * cos(alpha_v * x) * sin(alpha_v * y) *
           cos(alpha_v * z);
  }

  Real dwdx(Real x, Real y, Real z) const {
    return -ampl_w * alpha_w * sin(alpha_w * x) * sin(alpha_w * y) *
           sin(alpha_w * z);
  }

  Real dwdy(Real x, Real y, Real z) const {
    return ampl_w * alpha_w * cos(alpha_w * x) * cos(alpha_w * y) *
           sin(alpha_w * z);
  }

  Real dwdz(Real x, Real y, Real z) const {
    return ampl_w * alpha_w * cos(alpha_w * x) * sin(alpha_w * y) *
           cos(alpha_w * z);
  }

  Real dpdx(Real x, Real y, Real z) const {
    return -ampl_p * alpha_p * sin(alpha_p * x) * sin(alpha_p * y) *
           cos(alpha_p * z);
  }

  Real dpdy(Real x, Real y, Real z) const {
    return ampl_p * alpha_p * cos(alpha_p * x) * cos(alpha_p * y) *
           cos(alpha_p * z);
  }

  Real dpdz(Real x, Real y, Real z) const {
    return -ampl_p * alpha_p * cos(alpha_p * x) * sin(alpha_p * y) *
           sin(alpha_p * z);
  }

  Real dTdx(Real x, Real y, Real z) const {
    return -ampl_t * alpha_t * sin(alpha_t * x) * sin(alpha_t * y) *
           cos(alpha_t * z);
  }

  Real dTdy(Real x, Real y, Real z) const {
    return ampl_t * alpha_t * cos(alpha_t * x) * cos(alpha_t * y) *
           cos(alpha_t * z);
  }

  Real dTdz(Real x, Real y, Real z) const {
    return -ampl_t * alpha_t * cos(alpha_t * x) * sin(alpha_t * y) *
           sin(alpha_t * z);
  }
};

TEST_F(MMS, keep) {
  typedef closures_dt<visc_suth_t, cond_suth_t, calorifically_perfect_gas_t>
      cls_t;
  cls_t cls;

  typedef keep_euler_t<false, false, 2, cls_t> keep2_t;
  typedef keep_euler_t<false, false, 4, cls_t> keep4_t;
  typedef keep_euler_t<false, false, 6, cls_t> keep6_t;

  keep2_t keep2;
  keep4_t keep4;
  keep6_t keep6;

  Array1D<Real, 0, 2> dx= {0.151, 0.151, 0.151};
  Array1D<Real, 0, 2> error2,error4,error6;

  for (int i=0; i<3; i++){
    error2(i) = test_keep_scheme<keep2_t, cls_t>(keep2, cls, dx);
    error4(i) = test_keep_scheme<keep4_t, cls_t>(keep4, cls, dx);
    error6(i) = test_keep_scheme<keep6_t, cls_t>(keep6, cls, dx);
    dx(0) = dx(0)/2;
    dx(1) = dx(1)/2;
    dx(2) = dx(2)/2;
  }

  printf("                 dx : %f  %f  %f \n",dx(0)*2*2, dx(0)*2, dx(0));
  printf("Max error 2nd order KEEP: %f  %f  %f \n",error2(0),error2(1),error2(2));
  printf("Max error 4th order KEEP: %f  %f  %f \n",error4(0),error4(1),error4(2));
  printf("Max error 6th order KEEP: %f  %f  %f \n\n",error6(0),error6(1),error6(2));

  printf("Linf error 2nd order KEEP: %f  %f \n",log2(error2(0)/error2(1)),log2(error2(1)/error2(2)));
  printf("Linf error 4th order KEEP: %f  %f \n",log2(error4(0)/error4(1)),log2(error4(1)/error4(2)));
  printf("Linf error 6th order KEEP: %f  %f \n",log2(error6(0)/error6(1)),log2(error6(1)/error6(2)));

  Real order2 = (log2(error2(0)/error2(1)) + log2(error2(1)/error2(2)))/2;
  Real order4 = (log2(error4(0)/error4(1)) + log2(error4(1)/error4(2)))/2;
  Real order6 = (log2(error6(0)/error6(1)) + log2(error6(1)/error6(2)))/2;
  
  EXPECT_GE(order2,2.0);
  EXPECT_GE(order4,4.0);
  EXPECT_GE(order6,6.0);
}

TEST_F(MMS, scheme2) {


}

// TEST_F(MMS, scheme3) {}

// TODO: test the public function eflux directly. However, this requires setup
// an amr grid. Like NALU wind. However, here the first aim is to have simplest
// unit tests. Vprimsmf[level]define(grids, dmap, NPRIM, NGHOST, MFInfo(),
// Factory()); void inline eflux(const Geometry& geom, const MFIter& mfi,
//                   const Array4<Real>& prims, const Array4<Real>& cons,
//                   const Array4<Real>& flx, const Array4<Real>& rhs,
//                   const closures& cls)

#endif