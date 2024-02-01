#ifndef HLLCMP_H_
#define HLLCMP_H_

///
/// \brief Template class for multiphase HLLC method
///
/// \param isOption1  Immersed boundary fluxes
///
/// ```
/// {rst}
/// Divergence split:
/// :math:`\frac{\partial f}{\partial x}\bigg|_i= \frac{\partial}{}`
///
/// Quadratic split:
/// :math:`\frac{\partial fg}{\partial x}\bigg|_i= \frac{\partial}{}`
///
/// Cubic split:
/// :math:`\frac{\partial fgh}{\partial x}\bigg|_i= \frac{\partial}{}`
/// ```
///

template <bool isOption1, typename cls_t>
class hllc_mp_t {
 public:
  AMREX_GPU_HOST_DEVICE
  hllc_mp_t() {}
  ~hllc_mp_t() {}


  void inline eflux(const Geometry& geom, const MFIter& mfi,
                    const Array4<Real>& prims, const Array4<Real>& flx,
                    const Array4<Real>& rhs,
                    const cls_t& cls) {
    const GpuArray<Real, AMREX_SPACEDIM> dxinv = geom.InvCellSizeArray();
    const Box& bx  = mfi.growntilebox(0);
    const Box& bxg = mfi.growntilebox(cls.NGHOST);
    const Box& bxgnodal = mfi.grownnodaltilebox(-1, 0); // [0,N+1]
    
    amrex::Print() << "eflux multiphase \n";
    exit(0);
    // in x
    // get qL and qR


    // 1.) compute physical fluxes F_L & F_R (Eq. 16) 

    // 2.) compute star state

    // 3.) prim2cons QL and QR to WL and WR

    // 4.) plug everything in and get F_HLLC 

    // 5.) F2G
}
  private:

};

#endif