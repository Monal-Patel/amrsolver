#ifndef IBM_H_
#define IBM_H_


#include <AMReX_Amr.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_FabArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_Derive.H>
#include <AMReX_StateDescriptor.H>
#include <AMReX_GpuContainers.H>
#include <prob.h>

// basic CGAL headers
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>


// CGAL headers for AABB tree for closest point
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>


// CGAL headers for AABB tree for surface data
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
// #include <CGAL/Unique_hash_map.h>
// #include <boost/unordered_map.hpp>
#include <boost/property_map/property_map.hpp>

// CGAL header for inout testing
#include <CGAL/Side_of_triangle_mesh.h>

using namespace amrex;

namespace IBM {
  // import options from prob.h ---------------------------
  constexpr int NIMPS = PROB::nimps;
  using closures=PROB::ProbClosures;
  //--------------------------------------------------------


  // CGAL types -------------------------------------------
  typedef CGAL::Simple_cartesian<Real> K2;
  typedef K2::Point_3 Point;
  typedef CGAL::Polyhedron_3<K2> Polyhedron;

  typedef K2::FT FT;
  typedef K2::Segment_3 Segment;
  typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
  typedef CGAL::AABB_traits<K2, Primitive> Traits;
  typedef CGAL::AABB_tree<Traits> Tree;
  typedef Tree::Point_and_primitive_id Point_and_primitive_id;

  typedef K2::Vector_3 Vector_CGAL;
  typedef boost::graph_traits<Polyhedron>::face_descriptor  face_descriptor;
  typedef CGAL::Side_of_triangle_mesh<Polyhedron, K2> inside_t;
  // CGAL types -------------------------------------------

  // surfdata holds the relevant surface data for each element.
  struct surfdata {
    Array<Real,NPRIM> state={0.0};
    Array<Real,AMREX_SPACEDIM> displace = {0.0};
  };

  struct gpData_t {

      gpData_t() {};
      ~gpData_t() {};

      // CPU only attributes
      int ngps;
      Vector<Point_and_primitive_id> closest_cgal; // closest surface point (ib point) and face

      // GPU/CPU attributes
      Gpu::ManagedVector<Array1D<int,0,AMREX_SPACEDIM-1>> gp_ijk;
      Gpu::ManagedVector<Array1D<Real,0,AMREX_SPACEDIM-1>> normal,tangent1,tangent2, ib_xyz;
      Gpu::ManagedVector<Real> disGP,disIM; 
      Gpu::ManagedVector<int> geomIdx; 

      Gpu::ManagedVector<Array2D<Real,0,NIMPS-1,0,AMREX_SPACEDIM-1>> imp_xyz;
      Gpu::ManagedVector<Array2D<int, 0,NIMPS-1,0,AMREX_SPACEDIM-1>> imp_ijk;
      Gpu::ManagedVector<Array2D<Real,0,NIMPS-1,0,7>> imp_ipweights;
      Gpu::ManagedVector<Array3D<int,0,NIMPS-1,0,7,0,AMREX_SPACEDIM-1>> imp_ip_ijk;
      // for imp1 [(i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1),
      //  ... ]


  };

  // IBFab holds the solid point and ghost point boolean arrays
  class IBFab: public BaseFab<bool> 
  {
    public:
      // attributes //
      gpData_t gpData; // a single structure which holds GP data in vectors which can be resized on CPU or GPU.

      // constructors and destructors //
      // using Box
      explicit inline IBFab (const Box& b, int ncomp, bool alloc=true, 
                      bool shared=false, Arena* ar = nullptr) : BaseFab<bool>(b,ncomp,alloc,shared,ar) {};
      // using IBFab
      explicit inline IBFab (const IBFab& rhs, MakeType make_type, int scomp, 
                      int ncomp) : BaseFab<bool>(rhs,make_type,scomp,ncomp) {};

      // IBFab (IBFab&& rhs) noexcept = default;
      // IBFab& operator= (IBFab&&) noexcept = default;

      // IBFab (const IBM::IBFab&) {};
      // IBFab& operator= (const IBFab&) = delete;

     ~IBFab () {};

  };

  // IBMultiFab holds an array of IBFab on a level
  class IBMultiFab: public FabArray<IBFab> {
    public:
      // constructor from BoxArray and DistributionMapping
      explicit inline IBMultiFab ( const BoxArray& bxs, 
                            const DistributionMapping& dm, 
                            const int nvar, 
                            const int ngrow, 
                            const MFInfo& info = MFInfo(), 
                            const FabFactory<IBFab>& factory = DefaultFabFactory<IBFab>()) :
                        FabArray<IBFab>(bxs,dm,nvar,ngrow,info,factory) {};

      // move constructor
      IBMultiFab (IBMultiFab&& rhs) noexcept 
      : FabArray<IBFab>(std::move(rhs)) {
#ifdef AMREX_MEM_PROFILING
        ++num_multifabs;
        num_multifabs_hwm = std::max(num_multifabs_hwm, num_multifabs);
#endif
      }; 

      // destructor
      ~IBMultiFab () {};

      void inline copytoRealMF(MultiFab &mf, int ibcomp, int mfcomp) {
        for (MFIter mfi(*this,false); mfi.isValid(); ++mfi) {

            // const Box& ibbox = mfi.fabbox(); // box with ghost points
            const Box& ibbox = mfi.validbox(); // box without ghost points

            IBM::IBFab &ibfab = this->get(mfi);
            FArrayBox &realfab = mf.get(mfi);
            Array4<bool> ibMarkers = ibfab.array(); // boolean array
            Array4<Real> realfield = realfab.array(); // real array

            amrex::ParallelFor(ibbox,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              realfield(i,j,k,mfcomp)   = ibMarkers(i,j,k,ibcomp);
              realfield(i,j,k,mfcomp+1) = ibMarkers(i,j,k,ibcomp+1);
            });
        }
      };
    
    protected:

  };


  // interpolations ////////////////////////////////////////////////////////////
  class trilinear_interp_t{
    public:
      // note runs on CPU
      void computeIPweights(Array2D<Real,0,NIMPS-1,0,7>&weights,Array3D<int,0,NIMPS-1,0,7,0,AMREX_SPACEDIM-1>&ip_ijk, Array2D<Real,0,NIMPS-1,0,AMREX_SPACEDIM-1>&imp_xyz, Array2D<int,0,NIMPS-1,0,AMREX_SPACEDIM-1>& imp_ijk, const GpuArray<Real,AMREX_SPACEDIM>& prob_lo, GpuArray<Real,AMREX_SPACEDIM>& dxyz, const Array4<bool> ibFab) {
      // Array2D<int,0,7,0,AMREX_SPACEDIM-1> indexCube={{0,0,0},{0,1,0},{1,1,0},{1,0,0},{0,0,1},{0,1,1},{1,1,1},{1,0,1}}; // index cube for image point interpolation
      // Anti-clockwise order k=0 plane first, then k=1 plane.
      for (int iim=0; iim<NIMPS; iim++) {
        int i = imp_ijk(iim,0); int j = imp_ijk(iim,1); int k = imp_ijk(iim,2); 
        // note xl,xr, ...etc do not have prob_lo added to them. This does not matter as we only need the relative distances between the points.
        Real xl = prob_lo[0] + (Real(i)+0.5_rt) * dxyz[0];  // bottom left corner of cell
        Real xr = xl + dxyz[0];
        Real yl = prob_lo[1] + (Real(j)+0.5_rt) * dxyz[1];
        Real yr = yl + dxyz[1];
        Real zl = prob_lo[2] + (Real(k)+0.5_rt) * dxyz[2];
        Real zr = zl + dxyz[2];

        Real xd =  (imp_xyz(iim,0) - xl )/(xr-xl);
        Real yd =  (imp_xyz(iim,1) - yl )/(yr-yl);
        Real zd =  (imp_xyz(iim,2) - zl )/(zr-zl);

        int sumfluid = 0;
        Real sumweights = 0.0_rt;
        // zd = 0
        int ii = i;
        int jj = j;
        int kk = k;
        int iip= 0;
        int fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = (1.0_rt - xd) *(1.0_rt - yd)*(1.0_rt-zd)*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,0);

        ii = i + 0;
        jj = j + 1;
        kk = k + 0;
        iip= 1;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = (1.0_rt - xd) *yd*(1.0_rt-zd)*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,iip);

        ii = i + 1;
        jj = j + 1;
        kk = k + 0;
        iip= 2;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = xd*yd*(1.0_rt-zd)*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,iip);

        ii = i + 1;
        jj = j + 0;
        kk = k + 0;
        iip= 3;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = xd*(1.0_rt - yd)*(1.0_rt-zd)*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,iip);

        // zd = 2
        ii = i + 0;
        jj = j + 0;
        kk = k + 1;
        iip= 4;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = (1.0_rt - xd) *(1.0_rt - yd)*zd*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,iip);

        ii = i + 0;
        jj = j + 1;
        kk = k + 1;
        iip= 5;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = (1.0_rt - xd) *yd*zd*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,5);

        ii = i + 1;
        jj = j + 1;
        kk = k + 1;
        iip= 6;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = xd*yd*zd*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,iip);

        ii = i + 1;
        jj = j + 0;
        kk = k + 1;
        iip= 7;
        fluid = !ibFab(ii,jj,kk, 0);
        weights(iim,iip) = xd*(1.0_rt - yd)*zd*fluid;
        ip_ijk(iim,iip,0) = ii; ip_ijk(iim,iip,1) = jj; ip_ijk(iim,iip,2) = kk;
        sumfluid += fluid; sumweights += weights(iim,iip);

        AMREX_ASSERT_WITH_MESSAGE( sumfluid >= 2,"Less than 2 interpolation points are fluid points");

        // re-normalise
        for (int ll=0; ll<8; ll++) {
          weights(iim,ll) = weights(iim,ll)/sumweights;
        }

        AMREX_ASSERT_WITH_MESSAGE(std::abs(weights(iim,0) + weights(iim,1) + weights(iim,2) + weights(iim,3) + weights(iim,4) + weights(iim,5) + weights(iim,6) + weights(iim,7) - Real(1.0)) < Real(1.e-9),"Interpolation point weights do not sum to 1.0");
      }
    }
  };

  class wlsq_interp_t{

  };
  //////////////////////////////////////////////////////////////////////////////

  // extrapolations ////////////////////////////////////////////////////////////
  class linear_extrap_t{
    public:
      // Taylor expansion around IB point
      AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
      void extrapolate(Array2D<Real,0,NIMPS+1,0,NPRIM-1>& stateNormal, Real dgp, Real dim) {
        Real sgn_dgp = -dgp ; // negative sign as taylor expansion is around IB point, IM and GP are in opposite directions
        for (int kk=0; kk<NPRIM; kk++) {
          // Linear
          Real c1 = stateNormal(1,kk);
          Real c2 = (stateNormal(2,kk) - stateNormal(1,kk))/dim;
          stateNormal(0,kk) = c1 + c2*sgn_dgp;
        }
      }
  };
  
  class linear_limited_extrap_t {
    // Linear + Van Leer limiter
    // Real c1 = state(1,kk);
    // Real ratio  = (state(3,kk)-state(2,kk) )/( state(2,kk)-state(1,kk) + 1.0e-12);
    // Real phi    = (ratio + std::abs(ratio))/(1.0_rt + std::abs(ratio));
    // Real c2     = phi*(state(2,kk)-state(1,kk))/dim;
    // state(0,kk) = c1 + c2*sgn_dgp;

    // Real c1    =  state(1,kk);
    // Real c2    =  (-1.5_rt*state(1,kk) + 2.0_rt*state(2,kk) - 0.5_rt*state(3,kk))/sgn_dgp;
    // Real c3    =  (state(1,kk) - 2.0_rt*state(2,kk) + state(3,kk))/sgn_dgp/sgn_dgp;
    // state(0,kk) = c1 + c2*dim + c3*dim*dim/2;
  };

  //////////////////////////////////////////////////////////////////////////////

  // wall model ////////////////////////////////////////////////////////////////
  class algebraic_wm_t{
    void ComputeGPStateWM() {Abort("algebraic wall model not implemented yet");}
  };

  class ode_wm_t{

  };

  class no_wm_t{
    public:

    // TODO: pass boundary conditions
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void computeIB(Array2D<Real,0,NIMPS+1,0,NPRIM-1>& primsNormal, const closures& cls) {
      // no-slip velocity (in local coordinates)
      primsNormal(1,QU) = 0.0_rt; // un
      primsNormal(1,QV) = 0.0_rt; //primsNormal(2,QV); // ut1
      primsNormal(1,QW) = 0.0_rt; //primsNormal(2,QW); // ut2
      // zerograd temperature and pressure
      primsNormal(1,QPRES) = 2.0_rt*(2.0_rt*primsNormal(2,QPRES) - 0.5_rt*primsNormal(3,QPRES))/3.0_rt;
      primsNormal(1,QT)    = 500.0_rt; //primsNormal(2,QT);
      // ensure thermodynamic consistency
      primsNormal(1,QRHO)  = primsNormal(1,QPRES)/(primsNormal(1,QT)*cls.Rspec); 
    }

  };
  //////////////////////////////////////////////////////////////////////////////

 
  // scheme selection
  template <class wm, class interp, class extrap>
  class gp_scheme_tt: public wm, public interp, public extrap {
    public:

    // general interpolation routine -- over a given stencil points and weights
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
    void interpolateIMs(const Array3D<int,0,NIMPS-1,0,7,0,AMREX_SPACEDIM-1>& imp_ip_ijk, const Array2D<Real,0,2,0,7>& ipweights, const Array4<Real>& prims, Array2D<Real,0,NIMPS+1,0,NPRIM-1>& primsNormal){
      // for each image point
      for (int iim=0; iim<NIMPS; iim++) {
        // for each IP point
        for (int iip=0; iip<8; iip++) {
          int ii = imp_ip_ijk(iim,iip,0);
          int jj = imp_ip_ijk(iim,iip,1);
          int kk = imp_ip_ijk(iim,iip,2);
          // for each primitive variable
          for (int nn=0; nn<NPRIM; nn++) {
            // GP (jj=0),IB (jj=1), IM1 (jj=2),IM2 (jj=3)...
            primsNormal(iim+2,nn) += prims(ii,jj,kk,nn)*ipweights(iim,iip);
          }
        }
      }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
    void global2local( int iip, Array2D<Real,0,NIMPS+1,0,NPRIM-1>& primsNormal, const Array1D<Real,0,AMREX_SPACEDIM-1>& norm, const Array1D<Real,0,AMREX_SPACEDIM-1>& tan1, const Array1D<Real,0,AMREX_SPACEDIM-1>& tan2) {
      Array1D<Real,0,AMREX_SPACEDIM-1> vel;
      vel(0) = primsNormal(iip,QU); vel(1) = primsNormal(iip,QV); vel(2) = primsNormal(iip,QW);

      primsNormal(iip,QU) = vel(0)*norm(0) + vel(1)*norm(1) + vel(2)*norm(2);
      primsNormal(iip,QV) = vel(0)*tan1(0) + vel(1)*tan1(1) + vel(2)*tan1(2);
      primsNormal(iip,QW) = vel(0)*tan2(0) + vel(1)*tan2(1) + vel(2)*tan2(2);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
    void local2global (int jj, Array2D<Real,0,NIMPS+1,0,NPRIM-1>& primsNormal, const Array1D<Real,0,AMREX_SPACEDIM-1>& norm, const Array1D<Real,0,AMREX_SPACEDIM-1>& tan1, const Array1D<Real,0,AMREX_SPACEDIM-1>& tan2) {
      
      Array1D<Real,0,AMREX_SPACEDIM-1> vel;
      vel(0) = primsNormal(jj,QU); vel(1) = primsNormal(jj,QV); vel(2) = primsNormal(jj,QW);
      primsNormal(jj,QU) = vel(0)*norm(0) + vel(1)*tan1(0) + vel(2)*tan2(0);
      primsNormal(jj,QV) = vel(0)*norm(1) + vel(1)*tan1(1) + vel(2)*tan2(1);
      primsNormal(jj,QW) = vel(0)*norm(2) + vel(1)*tan1(2) + vel(2)*tan2(2);
    }

  };

  using gp_scheme_t = gp_scheme_tt<no_wm_t, trilinear_interp_t, linear_extrap_t>;
  //////////////////////////////////////////////////////////////////////////////

  // IB is the main class. It holds an array of IBMultiFab, one for each AMR level; and it also holds the geometry
  class ibm_t: public gp_scheme_t {
    public:
      // attributes //
      //image point distance per level
      Vector<Real> disIM; 

      // need to save this as Amr protects this -- set in ib.init(...)
      int MAX_LEVEL = 0;   

      // number of ghost points in IBMultiFab -- set in ib.init(...)
      int NGHOST_IB = 0; 

      // number of variables in IBMultiFab
      const int NVAR_IB  = 2; 

      // number of geometries
      int NGEOM=1; 

      // pointer to Amr class instance
      Amr* pamr;

      // vector of refinement ratio per level in each direction
      Vector<IntVect> ref_ratio;

      // vector of cell sizes per level in each direction
      Vector<GpuArray<Real,AMREX_SPACEDIM>> cellSizes;

      // Immersed boundary MultiFab array
      Vector<IBMultiFab*> ibMFa; 

      // Level set MultiFab array
      Vector<MultiFab> lsMFa;

      // IB explicit geometry
      Vector<Polyhedron> VGeom;  

      // AABB tree
      Vector<Tree*> VtreePtr;

      // Instead of std::map you may use std::unordered_map, boost::unordered_map
      // or CGAL::Unique_hash_map
      // CGAL::Unique_hash_map<face_descriptor,Vector> fnormals;
      // boost::unordered_map<vertex_descriptor,Vector> vnormals;
      // face element normals
      Vector<std::map<face_descriptor,Vector_CGAL>> Vfnormals;

      // face state data map
      // std::map<face_descriptor,surfdata> face2state;
      // std::map stores information in a binary tree, it has log(N) complexity for key-value pair insertion and value retrieval for a key.
      // could also fit normals into this -- however might need to modify compute_normals routine

      // face displacement map
      // std::map<face_descriptor,Vector_CGAL> fdisplace;
      // Vector or Real[3] don't work

      // in out testing function
      Vector<inside_t*> VInOutFunc;

      explicit ibm_t ();

      ~ibm_t ();
      

      // methods //
      void init (Amr* pointer_amr, const int nghost);

      void static compute_plane_equations( Polyhedron::Facet& f);

      void buildMFs (const BoxArray& bxa, const DistributionMapping& dm, int lev);

      void destroyMFs (int lev);

      void readGeom();

      void computeMarkers (int lev);

      void initialiseGPs(int lev);

      void computeGPs( int lev, MultiFab& consmf, MultiFab& primsmf, IBMultiFab& ibmf, const closures& cls);

      void moveGeom();

      void computeSurf(int lev);

      void applyBCs();

    private:

  };

  // declare main IB class instance

  inline ibm_t ibm;

}
#endif