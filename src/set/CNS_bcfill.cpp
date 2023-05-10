
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
#include <CNS_index_macros.H>
#include <CNS.H>
using namespace amrex;


// 1st order accurate for 3 ghost points
AMREX_GPU_DEVICE inline void dirichlet ( Array2D<Real,0,NCONS,0,5>& sten, int n, Real value) noexcept {
  // reflect
  Real delta = 2*value;
  sten(n,0) = delta-sten(n,5);
  sten(n,1) = delta-sten(n,4);
  sten(n,2) = delta-sten(n,3);
}

// 1st order accurate for 3 ghost points
AMREX_GPU_DEVICE inline void neumann ( Array2D<Real,0,NCONS,0,5>& sten, int n,Real grad, Real dx) noexcept {
  // linear copy
  Real delta = dx*grad;
  sten(n,0) = sten(n,5) - delta;
  sten(n,1) = sten(n,4) - delta;
  sten(n,2) = sten(n,3) - delta;
}

// This is called per boundary point
struct CnsFillExtDir
{
    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& data,
                     const int /*dcomp*/, const int /*numcomp*/,
                     GeometryData const& geom, const Real /*time*/,
                     const BCRec* bcr, const int /*bcomp*/,
                     const int /*orig_comp*/) const
        {

          Array2D<Real,0,NCONS,0,5> prims;

          int nghost = 3;
          int i = iv[0]; int j = iv[1]; int k = iv[2];
          int jjadd,jj,jstart;
          int dir = 1;
          const Real dy = geom.CellSize(dir);
          Real rho,rhoinv,ux,uy,uz,rhoke,rhoei,p,T;

          bool notcorner= iv[0] >= geom.Domain().smallEnd(0) 
          && iv[0] <= geom.Domain().bigEnd(0)
          && iv[2] >= geom.Domain().smallEnd(2) 
          && iv[2] <= geom.Domain().bigEnd(2);

          if (notcorner) {
            // y=0 boundary/plane
            if (iv[dir]==geom.Domain().smallEnd(dir)-1) {
              jstart = -nghost;
              jjadd  = 1;
            }
            // y=ymax boundary/plane
            else if (iv[dir]==geom.Domain().bigEnd(dir)+1) {
              jstart = geom.Domain().bigEnd(dir) + nghost;
              jjadd  = -1;
            }
            else {
              /// DUMMY
              // p   = 3000;//prims(QPRES,count);
              // T   = 300;//prims(QT,count);
              // rho = p/(CNS::d_parm->Rspec * T);
              // ux  = 0.0;
              // uy  = 0.0;
              // uz  = 0.0;
              // rhoke  = Real(0.5)*rho*(ux*ux + uy*uy + uz*uz);
              // rhoei  = p/(CNS::d_parm->eos_gamma - Real(1.0));
              // data(i,j,k,URHO) = rho;
              // data(i,j,k,UMX)  = rho*ux;
              // data(i,j,k,UMY)  = rho*uy;
              // data(i,j,k,UMZ)  = rho*uz;
              // data(i,j,k,UET)  = rhoke+rhoei;
              
              return ;}

            // convert to primitive vars (rho,u,v,w,T,P(T,rho))
            jj = jstart;
            for (int count=0; count<nghost*2; count++ ) {
              rho    = data(i,jj,k,URHO);
              rhoinv = Real(1.0)/rho;
              ux     = data(i,jj,k,UMX)*rhoinv;
              uy     = data(i,jj,k,UMY)*rhoinv;
              uz     = data(i,jj,k,UMZ)*rhoinv;
              rhoke  = Real(0.5)*rho*(ux*ux + uy*uy + uz*uz);
              rhoei  = (data(i,jj,k,UET) - rhoke);
              p      = (CNS::d_parm->eos_gamma - Real(1.0))*rhoei;

              prims(QRHO,count)  = rho;
              prims(QU,count)    = ux;
              prims(QV,count)    = uy;
              prims(QW,count)    = uz;
              prims(QPRES,count) = p;
              prims(QT,count)    = p/(rho*CNS::d_parm->Rspec);
              jj = jj + jjadd;
            }

            // u,v,w,T,P
            dirichlet(prims,QU,Real(0.0)); 
            // neumann  (prims,QU,Real(0.0),dy); 
            dirichlet(prims,QV,Real(0.0)); 
            dirichlet(prims,QW,Real(0.0)); 
            dirichlet(prims,QT,CNS::d_prob_parm->Tw); 
            neumann  (prims,QPRES,Real(0.0),dy); 

            // convert back to conservative vars
            // compute rho
            jj = jstart;
            for (int count=0; count<nghost; count++ ) {
              p   = prims(QPRES,count);
              T   = prims(QT,count);
              rho = p/(CNS::d_parm->Rspec * T);
              ux  = prims(QU,count);
              uy  = prims(QV,count);
              uz  = prims(QW,count);
              rhoke  = Real(0.5)*rho*(ux*ux + uy*uy + uz*uz);
              rhoei  = p/(CNS::d_parm->eos_gamma - Real(1.0));

              data(i,jj,k,URHO) = rho;
              data(i,jj,k,UMX)  = rho*ux;
              data(i,jj,k,UMY)  = rho*uy;
              data(i,jj,k,UMZ)  = rho*uz;
              data(i,jj,k,UET)  = rhoke+rhoei;

              jj = jj + jjadd;
            }

          }

        }
};

// bx                  : Cells outside physical domain and inside bx are filled.
// data, dcomp, numcomp: Fill numcomp components of data starting from dcomp.
// bcr, bcomp          : bcr[bcomp] specifies BC for component dcomp and so on.
// scomp               : component index for dcomp as in the descriptor set up in CNS::variableSetUp.
// This is called once per fab (if fab has domain boundary and NCONS=NVAR boundary to set)
void cns_bcfill (Box const& bx, FArrayBox& data,
                 const int dcomp, const int numcomp,
                 Geometry const& geom, const Real time,
                 const Vector<BCRec>& bcr, const int bcomp,
                 const int scomp)
{
  //////////////////////////// ALTERNATIVE POSSIBLE IMPLEMENTATION /////////////
    // const Real*    dx          = geom.CellSize();
    // // ymin boundary
    // if (bx.smallEnd(1) < geom.Domain().smallEnd(1)) {
    //   // fill ymin boundary
    // }
    // if (bx.bigEnd(1) > geom.Domain().bigEnd(1)) {
    //   // fill ymax boundary
    // }
  //////////////////////////////////////////////////////////////////////////////

    // Currently we assume ymax and ymin BC is wall 
    // GpuBndryFuncFab class operates on all boundaries of the fab. It calls CnsFillExtDir for each ghost point ijk.
    GpuBndryFuncFab<CnsFillExtDir> gpu_bndry_func(CnsFillExtDir{});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}
