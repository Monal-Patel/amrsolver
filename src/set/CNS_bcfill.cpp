
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
#include <CNS_index_macros.H>
#include <CNS.H>
using namespace amrex;


// 2st order accurate for 3 ghost points
AMREX_GPU_DEVICE inline void dirichlet ( auto& sten, int ivar, int nghost, Real value) noexcept {
  // reflect
  Real delta = 2*value;
  for (int j=0;j<nghost;j++) {
    sten(ivar,j) = delta - sten(ivar,2*nghost-1-j);
  }
}

AMREX_GPU_DEVICE inline void zerograd_pres ( auto& sten, int ivar, int nghost) noexcept {
  // linear copy
  for (int j=0;j<nghost;j++) {
    sten(ivar,j) = sten(ivar,nghost);
  }
}

// This is called per boundary point
struct CnsFillExtDir
{
    // create pointers to device (for gpu) parms
    ProbParm* lprobparm = CNS::d_prob_parm;
    Parm*     lparm     = CNS::d_parm;
    BCRec* l_phys_bc    = CNS::d_phys_bc;
    int nghost          = CNS::NGHOST;

    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& data,
                     const int /*dcomp*/, const int /*numcomp*/,
                     GeometryData const& geom, const Real /*time*/,
                     const BCRec* bcr, const int /*bcomp*/,
                     const int /*orig_comp*/) const
        {

        // IMPROVEMENT TODO: Avoid if statement by having different options for user defined BCs in cns_bcfill
        if (l_phys_bc->lo(1)==6 || l_phys_bc->hi(1)==6) {
          Array2D<Real,0,NPRIM-1,0,5> prims;

          // int nghost = 3;
          int i = iv[0]; int j = iv[1]; int k = iv[2];
          int jjadd,jj,jstart;
          int dir = 1;
          const Real dy = geom.CellSize(dir);
          Real rho,rhoinv,ux,uy,uz,rhoke,rhoei,p,T;

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
          else { return ;}

            // convert domain points near boundary to primitive vars (rho,u,v,w,T,P(T,rho)), in preparation to fill ghost points in primitive vars
            jj = jstart + jjadd*nghost; // only convert in domain
            for (int count=nghost; count<2*nghost; count++ ) {
              rho    = data(i,jj,k,URHO);
              rhoinv = Real(1.0)/rho;
              ux     = data(i,jj,k,UMX)*rhoinv;
              uy     = data(i,jj,k,UMY)*rhoinv;
              uz     = data(i,jj,k,UMZ)*rhoinv;
              rhoke  = Real(0.5)*rho*(ux*ux + uy*uy + uz*uz);
              rhoei  = (data(i,jj,k,UET) - rhoke);
              p      = (lparm->eos_gamma - Real(1.0))*rhoei;

              prims(QRHO,count)  = rho;
              prims(QU,count)    = ux;
              prims(QV,count)    = uy;
              prims(QW,count)    = uz;
              prims(QPRES,count) = p;
              prims(QT,count)    = p/(rho*lparm->Rspec);
              jj = jj + jjadd;
            }

            // apply BC (fill u,v,w,T,P)
            dirichlet(prims,QU,nghost,Real(0.0)); 
            // neumann  (prims,QU,Real(0.0),dy); 
            dirichlet(prims,QV,nghost,Real(0.0)); 
            dirichlet(prims,QW,nghost,Real(0.0)); 
            dirichlet(prims,QT,nghost,lprobparm->Tw); 
            zerograd_pres(prims,QPRES,nghost); 

            // convert ghost points to conservative vars
            // compute rho
            jj = jstart;
            for (int count=0; count<nghost; count++ ) {
              p   = prims(QPRES,count);
              T   = prims(QT,count);
              rho = p/(lparm->Rspec * T);
              ux  = prims(QU,count);
              uy  = prims(QV,count);
              uz  = prims(QW,count);
              rhoke  = Real(0.5)*rho*(ux*ux + uy*uy + uz*uz);
              rhoei  = p/(lparm->eos_gamma - Real(1.0));

              // if (rho==Real(0.0)) {
              //   printf("%d %f %f %f",count,p,rho,T);
              //   amrex::Abort("rho is 0.0");
              //   };

              // if (i==0 && k==0) {
              // printf("fill bc i=%d,jj=%d,k=%d \n",i,jj,k);
              // printf("fill bc rho %f \n \n",rho);
              // }

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
    Gpu::streamSynchronize();
}
