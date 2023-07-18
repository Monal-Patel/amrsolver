#include <IBM.H>
#include <CGAL/Side_of_triangle_mesh.h>
#include <AMReX_ParmParse.H>

using namespace amrex;
using namespace IBM;

IBFab::IBFab (const Box& b, int ncomp, bool alloc, bool shared, Arena* ar)    
              : BaseFab<bool>(b,ncomp,alloc,shared,ar) {}
IBFab::IBFab (const IBFab& rhs, MakeType make_type, int scomp, int ncomp) 
              : BaseFab<bool>(rhs,make_type,scomp,ncomp) {}
IBFab::~IBFab () { }

IBMultiFab::IBMultiFab ( const BoxArray& bxs, const DistributionMapping& dm, 
                        const int nvar, const int ngrow, const MFInfo& info, 
                        const FabFactory<IBFab>& factory )  :
                        FabArray<IBFab>(bxs,dm,nvar,ngrow,info,factory) {}
IBMultiFab::~IBMultiFab () {}

// for a single level
void IBMultiFab::copytoRealMF(MultiFab &mf, int ibcomp, int mfcomp) {

  for (MFIter mfi(*this,false); mfi.isValid(); ++mfi) {

    // const Box& ibbox = mfi.fabbox(); // box with ghost points
    const Box& ibbox = mfi.validbox(); // box without ghost points

    IBM::IBFab &ibfab = this->get(mfi);
    FArrayBox &realfab = mf.get(mfi);
    Array4<bool> ibMarkers = ibfab.array(); // boolean array
    Array4<Real> realfield = realfab.array(); // real array

    // assert that box sizes is same - TODO

    // const int *lo = fab.loVect();
    // const int *hi = fab.hiVect();
    // amrex::Print() << lo[0] << " " << lo[1] << " " << lo[2] << std::endl;
    // amrex::Print() << hi[0] << " " << hi[1] << " " << hi[2] << std::endl;

    amrex::ParallelFor(ibbox,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      realfield(i,j,k,mfcomp)   = ibMarkers(i,j,k,ibcomp);
      realfield(i,j,k,mfcomp+1) = ibMarkers(i,j,k,ibcomp+1);
    });
  }
}

// constructor and destructor
IB::IB (){}
IB::~IB () { delete treePtr;}

void IB::setMaxLevel(int max_lev) {
  // parent->finestLevel()

};

// initialise IB
void IB::init(Amr* pointer_amr, const int nghost) {


  IB::nghost_ib = nghost;
  IB::pamr = pointer_amr ; // store pointer to main Amr class object's instance
  IB::ref_ratio = pamr->refRatio();
  IB::max_level = pamr->maxLevel();
  IB::mfa.resize(max_level + 1);

  // TODO make the next 15 lines simpler
  IB::cellSizes.resize(IB::max_level+1);
  IB::dimp.resize(IB::max_level+1);
  cellSizes[0] = pamr->Geom(0).CellSizeArray();
  for (int i=1;i<=IB::max_level;i++) {
    for (int j=0;j<=AMREX_SPACEDIM-1;j++) {
      cellSizes[i][j] = cellSizes[i-1][j]/ref_ratio[i-1][j];
    }
  }

  // TODO make distance ip a parameter
  for (int i=0;i<=IB::max_level;i++) {
    IB::dimp[i] = 0.6_rt*sqrt(cellSizes[i][0]*cellSizes[i][0] 
    + cellSizes[i][1]*cellSizes[i][1] + cellSizes[i][2]*cellSizes[i][2]);
  }

  // read geometry
  readGeom();
}
// create IBMultiFabs at a level and store pointers to it
void IB::buildIBMultiFab (const BoxArray& bxa, const DistributionMapping& dm, int lev) {
  mfa.at(lev) = new IBMultiFab(bxa,dm,nvar_ib,nghost_ib);
}

void IB::destroyIBMultiFab (int lev) {
  if (!mfa.empty()) {
      delete mfa.at(lev);
  }
}

void IB::computeMarkers (int lev) {

  CGAL::Side_of_triangle_mesh<Polyhedron, K2> inside(IB::geom);

  IBMultiFab *mfab = mfa.at(lev);
  int nhalo = mfab->nGrow(0); // assuming same number of ghost points in all directions
  for (MFIter mfi(*mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &fab = mfab->get(mfi);
    const int *lo = fab.loVect();
    const int *hi = fab.hiVect();

    fab.setVal(false); // initialise sld and ghs to false
    Array4<bool> ibMarkers = fab.array(); // boolean array

    // compute sld markers (including at ghost points) - cannot use ParallelFor - CGAL call causes problems
    for (int k = lo[2]; k <= hi[2]; ++k) {
    for (int j = lo[1]; j <= hi[1]; ++j) {
    for (int i = lo[0]; i <= hi[0]; ++i) {
      Real x=(0.5 + i)*cellSizes[lev][0];
      Real y=(0.5 + j)*cellSizes[lev][1];
      Real z=(0.5 + k)*cellSizes[lev][2];
      Point gridpoint(x,y,z);
      CGAL::Bounded_side res = inside(gridpoint);

      if (res == CGAL::ON_BOUNDED_SIDE) { 
          // soild marker
          ibMarkers(i,j,k,0) = true;}
      else if (res == CGAL::ON_BOUNDARY) { 
        amrex::Print() << "Grid point on IB surface" << " " << std::endl;
        exit(0);
      }
    }}};

    // compute ghs markers ------------------------------
    fab.ngps =0;
    for (int k = lo[2]+nhalo; k <= hi[2]-nhalo; ++k) {
    for (int j = lo[1]+nhalo; j <= hi[1]-nhalo; ++j) {
    for (int i = lo[0]+nhalo; i <= hi[0]-nhalo; ++i) {
      bool ghost = false;
      if (ibMarkers(i,j,k,0)) {
        for (int l = -1; l<=1; ++l) {
          ghost = ghost || (!ibMarkers(i+l,j,k,0));
          ghost = ghost || (!ibMarkers(i,j+l,k,0));
          ghost = ghost || (!ibMarkers(i,j,k+l,0));
        }
        ibMarkers(i,j,k,1) = ghost; 
        fab.ngps = fab.ngps + ghost;
      }
    }}};

    // index gps ----------------------------------------
    // allocating space before preferred than using insert
    fab.gpArray.resize(fab.ngps); // create space for gps
    int ii=0;
    for (int k = lo[2]+nhalo; k <= hi[2]-nhalo; ++k) {
    for (int j = lo[1]+nhalo; j <= hi[1]-nhalo; ++j) {
    for (int i = lo[0]+nhalo; i <= hi[0]-nhalo; ++i) {
      if(ibMarkers(i,j,k,1)) {
        fab.gpArray[ii].idx[0] = i;
        fab.gpArray[ii].idx[1] = j;
        fab.gpArray[ii].idx[2] = k;
        ii += 1;
      }
    }}};
    // Print() <<ii << " " << fab.ngps <<std::endl;
  }
}


void IB::initialiseGPs (int lev) {

  IBMultiFab *mfab = mfa[lev];
  int nhalo = mfab->nGrow(0); // assuming same number of ghost points in all directions

  for (MFIter mfi(*mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &fab = mfab->get(mfi);

    for (int ii=0;ii<fab.ngps;ii++) {
      IntArray& idx = fab.gpArray[ii].idx;
      Real x=(0.5_rt + idx[0])*cellSizes[lev][0];
      Real y=(0.5_rt + idx[1])*cellSizes[lev][1];
      Real z=(0.5_rt + idx[2])*cellSizes[lev][2];
      Point gp(x,y,z);

      // closest surface point and face --------------------------
      fab.gpArray[ii].closest = treePtr->closest_point_and_primitive(gp);

      Point cp = fab.gpArray[ii].closest.first;
      Polyhedron::Face_handle face = fab.gpArray[ii].closest.second; // closest primitive id
      // Print() << "closest surface point: " << cp << std::endl;
      // Print() << "closest triangle: ( "
      //           << face->halfedge()->vertex()->point() << " , "
      //           << face->halfedge()->next()->vertex()->point() << " , "
      //           << face->halfedge()->next()->next()->vertex()->point()
      //           << " )" 
      //           << std::endl; 
      // Print() << "Normal " << fnormals[face] <<std::endl;
      // Print() << "cp-gp " << cp - gp << std::endl; // should be in the direction of normal
      // Print() << "Plane " << face->plane().a() << " " << face->plane().b() << " "  << face->plane().c() << " " << face->plane().d() << std::endl;
      // amrex::Print() << "------------------- " << std::endl;

      // IB point -------------------------------------------
      Vector_CGAL imp_gp(gp,cp);
      Real dibp = sqrt(imp_gp.squared_length());

      // IM point -------------------------------------------
      Vector<Array<Real,AMREX_SPACEDIM>>& imps = fab.gpArray[ii].imps; 
      Vector<Array<Real,AMREX_SPACEDIM>>& impscell = fab.gpArray[ii].impscell; 
      imps.resize(NUM_IMPS);
      impscell.resize(NUM_IMPS);
      impscell.resize(NUM_IMPS);

      for (int jj=0; jj<NUM_IMPS; jj++) {
        for (int kk=0; kk<AMREX_SPACEDIM; kk++) {
          imps[jj][kk] = cp[kk] + (jj+1)*IB::dimp[lev]*fnormals[face][kk];
          impscell[jj][kk] = floor(imps[jj][kk]/cellSizes[lev][kk] - 0.5_rt);
        }
      }

      Print() << "dibp, dimp (from gp) " << dibp << " " << IB::dimp[lev] << std::endl;
      for (int jj=0; jj<NUM_IMPS; jj++) {
        Print() << "imp" << jj+1 << " "<< fab.gpArray[ii].imps[jj] << std::endl;
        Print() << "impCell" << jj+1 << " "<< fab.gpArray[ii].impscell[jj] << std::endl;
      }
      
      // Interpolation points' weights
      Vector<Array<Real,8>>& ipweights = fab.gpArray[ii].ipweights; 

      for (int jj=0; jj<NUM_IMPS; jj++) {
        ipweights[jj] = computeIPweights();
      }

      exit(0);
    }
  }
}

Array<Real,8> IB::computeIPweights() {
  // ! tri-linear interpolation
  // xd =  (p1(1) - xl )/(xr-xl)
  // yd =  (p1(2) - yl )/(yr-yl)
  // zd =  (p1(3) - zl )/(zr-zl)

  // zd = 0
  // eta(1) = (1.0_wp - xd) *(1.0_wp - yd)*(1.0_wp-zd)
  // eta(2) = (1.0_wp - xd) *yd*(1.0_wp-zd)
  // eta(3) = xd*yd*(1.0_wp-zd)
  // eta(4) = xd*(1.0_wp - yd)*(1.0_wp-zd)

  // zd = 2
  // eta(5) = (1.0_wp - xd) *(1.0_wp - yd)*zd
  // eta(6) = (1.0_wp - xd) *yd*zd
  // eta(7) = xd*yd*zd
  // eta(8) = xd*(1.0_wp - yd)*zd
  Array<Real,8> weights;
  return weights;
}


void IB::computeGPs( int lev, MultiFab& S_new) {

  IBMultiFab *mfab = mfa[lev];
  int nhalo = mfab->nGrow(0); // assuming same number of ghost points in all directions

  for (MFIter mfi(*mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &fab = mfab->get(mfi);
    for (int ii=0;ii<fab.ngps;ii++) {
      IntArray& idx = fab.gpArray[ii].idx;

    // interpolate IM



    // interpolate IB


    // extrapolate
    }
  }

}


void IB::compute_plane_equations( Polyhedron::Facet& f) {
    Polyhedron::Halfedge_handle h = f.halfedge();
    f.plane() = Polyhedron::Plane_3( h->opposite()->vertex()->point(), 
		       h->vertex()->point(),
		       h->next()->vertex()->point());
};

void IB::readGeom() {

  ParmParse pp;
  std::string filename;
  pp.get("ib.filename",filename);

  namespace PMP = CGAL::Polygon_mesh_processing;

  if(!PMP::IO::read_polygon_mesh(filename, IB::geom) || CGAL::is_empty(IB::geom) || !CGAL::is_triangle_mesh(IB::geom))
  {
    std::cerr << "Invalid geometry filename" << std::endl;
    exit(1);
  }
  Print() << "----------------------------------" << std::endl;
  Print() << "Geometry " << filename << " read"<< std::endl;
  Print() << "----------------------------------" << std::endl;
  Print() << "Is geometry only made of triangles? " << geom.is_pure_triangle() << std::endl;
  Print() << "Number of facets " << geom.size_of_facets() << std::endl;

  // constructs AABB tree and computes internal KD-tree
  // data structure to accelerate distance queries
  treePtr = new Tree (faces(geom).first, faces(geom).second, geom);
  Print() << "AABB tree constructed" << std::endl;
  // treePtr->rebuild(faces(geom).first,faces(geom).second,geom)

  // Instead of std::map you may use std::unordered_map, boost::unordered_map
  // or CGAL::Unique_hash_map
  // CGAL::Unique_hash_map<face_descriptor,Vector> fnormals;
  // boost::unordered_map<vertex_descriptor,Vector> vnormals;
  PMP::compute_face_normals(geom, boost::make_assoc_property_map(fnormals));
  Print() << "Face normals computed" << std::endl;

  std::for_each( geom.facets_begin(), geom.facets_end(), compute_plane_equations);
  // plane class also computes orthogonal direction to the face. However, the orthogonal vector is not normalised.
}