#ifndef IBM_H_
#define IBM_H_

#include <AMReX_Amr.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_FabArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_Derive.H>
#include <AMReX_StateDescriptor.H>
#include <CNS_parm.H>
#include <CNS_index_macros.H>

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

using namespace amrex;

namespace IBM {

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

  // typedef Polyhedron::Vertex_iterator      vertexIter;
  // typedef Polyhedron::Face_iterator        faceIter;
  // CGAL types -------------------------------------------

  // surfdata holds the relevant surface data for each element.
  struct surfdata {
    Array<Real,NPRIM> state={0.0};
    Array<Real,AMREX_SPACEDIM> displace = {0.0};
  };

  // GP holds the relevant data for each ghost point.
  struct gp {

      // attributes
      Array<int , AMREX_SPACEDIM> idx={0};
      Point_and_primitive_id closest; // closest surface point (ib point) and face
      Real disGP; 
      Vector<Array<Real,AMREX_SPACEDIM>> imps; 
      Vector<Array<int,AMREX_SPACEDIM>> impscell; 
      Vector<Array<Real,8>> ipweights; 
      // for imp1 [(i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1),
      //  ... ]

  };

  // IBFab holds the solid point and ghost point boolean arrays
  class IBFab: public BaseFab<bool> 
  {
    public:
      // attributes //
      Vector<gp> gpArray;
      int ngps = 0;
      ///////////////

      // constructors and destructors //
      // using Box
      explicit IBFab (const Box& b, int ncomp, bool alloc=true, 
                      bool shared=false, Arena* ar = nullptr);
      // using IBFab
      explicit IBFab (const IBFab& rhs, MakeType make_type, int scomp, 
                      int ncomp);
     ~IBFab ();
     ///////////////////////////////////////////////////////////////////////////

      // methods //
      /////////////////////////////
  };

  // IBMultiFab holds an array of IBFab on a level
  class IBMultiFab: public FabArray<IBFab> {
    public:
      // constructor from BoxArray and DistributionMapping
      explicit IBMultiFab ( const BoxArray& bxs, 
                            const DistributionMapping& dm, 
                            const int nvar, 
                            const int ngrow, 
                            const MFInfo& info = MFInfo(), 
                            const FabFactory<IBFab>& factory = DefaultFabFactory<IBFab>());
      // destructor
      ~IBMultiFab ();

      void copytoRealMF(MultiFab &mf, int ibcomp, int mfcomp);
    protected:

  };

  // IB is the main class. It holds an array of IBMultiFab, one for each AMR level; and it also holds the geometry
  class IB {
    public:
      // attributes //
      Vector<Real> disIM; //image point distance per level

      int MAX_LEVEL = 0;   // need to save this as Amr protects this -- set in ib.init(...)
      int NGHOST_IB = 0; // number of ghost points in IBMultiFab -- set in ib.init(...)
      const int NVAR_IB  = 2; // number of variables in IBMultiFab
      const int NUM_IMPS = 2; // number of image points per ghost point

      Vector<amrex::Array<int,AMREX_SPACEDIM>> indexCube = { {0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}, {0,0,1}, {0,1,1}, {1,1,1}, {1,0,1}}; // index cube for image point interpolation
      // Anti-clockwise order k=0 plane first, then k=1 plane.


      // pointer to Amr class instance
      Amr*pamr;

      // vector of refinement ratio per level in each direction
      Vector<IntVect> ref_ratio;

      // vector of cell sizes per level in each direction
      Vector<GpuArray<Real,AMREX_SPACEDIM>> cellSizes;

      // IBMultiFab array
      Vector<IBMultiFab*> mfa; 

      // IB geometry
      Polyhedron geom;  

      // AABB tree
      Tree* treePtr;


    // Instead of std::map you may use std::unordered_map, boost::unordered_map
    // or CGAL::Unique_hash_map
    // CGAL::Unique_hash_map<face_descriptor,Vector> fnormals;
    // boost::unordered_map<vertex_descriptor,Vector> vnormals;
      // face element normals
      std::map<face_descriptor,Vector_CGAL> fnormals;
      // face state data map
      std::map<face_descriptor,surfdata> face2state;
      // std::map stores information in a binary tree, it has log(N) complexity for key-value pair insertion and value retrieval for a key.
      // could also fit normals into this -- however might need to modify compute_normals routine

      // face displacement map
      // std::map<face_descriptor,Vector_CGAL> fdisplace;
      // Vector or Real[3] don't work

      explicit IB ();
      ~IB ();
      

      // methods //
      void init (Amr* pointer_amr, const int nghost);

      void static compute_plane_equations( Polyhedron::Facet& f);

      void buildIBMultiFab (const BoxArray& bxa, const DistributionMapping& dm, int lev);

      void destroyIBMultiFab (int lev);

      void readGeom();

      void computeMarkers (int lev);

      void initialiseGPs(int lev);

      void computeGPs( int lev, MultiFab& consmf, MultiFab& primsmf, const Parm& parms);

      void moveGeom();

      void computeSurface();

    private:
      Array<Real,8> computeIPweights(Array<Real,AMREX_SPACEDIM>&imp, Array<int,AMREX_SPACEDIM>& impscell, GpuArray<Real,AMREX_SPACEDIM>& cellsize);

      void extrapolateGP(Array<Real,6>& stateGP, Array<Real,6>& stateIB, Vector<Array<Real,6>>& stateIMs, Real dgp, Real dim);

  };

  // declare main IB class instance
  inline IB ib;

}
#endif