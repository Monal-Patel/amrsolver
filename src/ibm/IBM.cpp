#include <IBM.H>
#include <CGAL/Side_of_triangle_mesh.h>
#include <AMReX_ParmParse.H>
#include <CNS_index_macros.H>

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

// void IB::setMaxLevel(int max_lev) {
//   // parent->finestLevel()

// };

// initialise IB
void IB::init(Amr* pointer_amr, const int nghost) {

  IB::NGHOST_IB = nghost;
  IB::pamr = pointer_amr ; // store pointer to main Amr class object's instance
  IB::ref_ratio = pamr->refRatio();
  IB::MAX_LEVEL = pamr->maxLevel();
  IB::ibMFa.resize(MAX_LEVEL + 1);
  IB::lsMFa.resize(MAX_LEVEL + 1);

  // TODO make the next 15 lines simpler
  IB::cellSizes.resize(IB::MAX_LEVEL+1);
  IB::disIM.resize(IB::MAX_LEVEL+1);
  cellSizes[0] = pamr->Geom(0).CellSizeArray();
  for (int i=1;i<=IB::MAX_LEVEL;i++) {
    for (int j=0;j<=AMREX_SPACEDIM-1;j++) {
      cellSizes[i][j] = cellSizes[i-1][j]/ref_ratio[i-1][j];
    }
  }

  // TODO make distance ip a parameter
  for (int i=0;i<=IB::MAX_LEVEL;i++) {
    IB::disIM[i] = 0.6_rt*sqrt(cellSizes[i][0]*cellSizes[i][0] 
    + cellSizes[i][1]*cellSizes[i][1] + cellSizes[i][2]*cellSizes[i][2]);
  }

  // read geometry
  readGeom();
}
// create IBMultiFabs at a level and store pointers to it
void IB::buildMFs (const BoxArray& bxa, const DistributionMapping& dm, int lev) {
  ibMFa.at(lev) = new IBMultiFab(bxa,dm,NVAR_IB,NGHOST_IB);
  lsMFa[lev].define(bxa,dm,1,NGHOST_IB);
}

void IB::destroyMFs (int lev) {
  if (!ibMFa.empty()) {
      delete ibMFa.at(lev);
  }
  if (!lsMFa.empty()) {
      lsMFa[lev].clear();
  }
}

void IB::computeMarkers (int lev) {

  CGAL::Side_of_triangle_mesh<Polyhedron, K2> inside(IB::geom);

  IBMultiFab& mfab = *ibMFa[lev];
  int nhalo = mfab.nGrow(0); // assuming same number of ghost points in all directions
  for (MFIter mfi(mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &fab = mfab.get(mfi);
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

  IBMultiFab& mfab = *ibMFa[lev];
  int nhalo = mfab.nGrow(0); // assuming same number of ghost points in all directions

  for (MFIter mfi(mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &fab = mfab.get(mfi);

    for (int ii=0;ii<fab.ngps;ii++) {
      IntArray& idx = fab.gpArray[ii].idx;
      Real x=(0.5_rt + idx[0])*cellSizes[lev][0];
      Real y=(0.5_rt + idx[1])*cellSizes[lev][1];
      Real z=(0.5_rt + idx[2])*cellSizes[lev][2];
      Point gp(x,y,z);

      // closest surface point and face --------------------------
      fab.gpArray[ii].closest = treePtr->closest_point_and_primitive(gp);

      // This closest point is between the face plane and the gp
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
      fab.gpArray[ii].disGP = sqrt(imp_gp.squared_length());
       

      // IM point -------------------------------------------
      Vector<Array<Real,AMREX_SPACEDIM>>& imps    = fab.gpArray[ii].imps; 
      Vector<Array<int,AMREX_SPACEDIM>>& impscell = fab.gpArray[ii].impscell; 
      imps.resize(NUM_IMPS);
      impscell.resize(NUM_IMPS);

      // find image point and the bottom left point closest to the image point
      //  In 2D, same idea in 3D.
      //     i,j+1 (2) ---------------------     i+1,j+1 (3)
      //     |                                  |
      //     |         P                        |
      //     |                                  |
      //     |                                  |
      //     i,j  (1) ----------------------      i+1,j  (4)
      for (int jj=0; jj<NUM_IMPS; jj++) {
        for (int kk=0; kk<AMREX_SPACEDIM; kk++) {
          imps[jj][kk] = cp[kk] + (jj+1)*IB::disIM[lev]*fnormals[face][kk];
          impscell[jj][kk] = floor(imps[jj][kk]/cellSizes[lev][kk] - 0.5_rt);
        }
      }
      // TODO assert that ip point within fab domain

      // temporary
      // Print() << "disGP, disIM (from IB) " << fab.gpArray[ii].disGP << " " << IB::disIM[lev] << std::endl;
      // for (int jj=0; jj<NUM_IMPS; jj++) {
      //   Print() << "imp" << jj+1 << " "<< fab.gpArray[ii].imps[jj] << std::endl;
      //   Print() << "impCell" << jj+1 << " "<< fab.gpArray[ii].impscell[jj] << std::endl;
      // }
      
      // Interpolation points' (ips) weights for each image point
      Vector<Array<Real,8>>& ipweights = fab.gpArray[ii].ipweights; 
      ipweights.resize(NUM_IMPS);

      for (int jj=0; jj<NUM_IMPS; jj++) {
        ipweights[jj] = computeIPweights(imps[jj], impscell[jj], cellSizes[lev]);
        // Real sum = 0.0;
        // Print() << "ipweights initGP" << jj+1 << std::endl;
        // for (int kk=0; kk<8; kk++) {
        //   sum += ipweights[jj][kk];
        //   Print() << kk+1 << " " << ipweights[jj][kk] << std::endl;
        // }
        // Print() << "sum " << sum << std::endl;
      }
    }
  }
}

Array<Real,8> IB::computeIPweights(Array<Real,AMREX_SPACEDIM>&imp, Array<int,AMREX_SPACEDIM>& impcell, GpuArray<Real,AMREX_SPACEDIM>& dxyz) {

  Array<Real,8> weights;

  // tri-linear interpolation
  Real xl = (Real(impcell[0])+0.5_rt) * dxyz[0];  // bottom left corner of cell
  Real xr = xl + dxyz[0];
  Real yl = (Real(impcell[1])+0.5_rt) * dxyz[1];
  Real yr = yl + dxyz[1];
  Real zl = (Real(impcell[2])+0.5_rt) * dxyz[2];
  Real zr = zl + dxyz[2];


  Real xd =  (imp[0] - xl )/(xr-xl);
  Real yd =  (imp[1] - yl )/(yr-yl);
  Real zd =  (imp[2] - zl )/(zr-zl);

  // zd = 0
  weights[0] = (1.0_rt - xd) *(1.0_rt - yd)*(1.0_rt-zd);
  weights[1] = (1.0_rt - xd) *yd*(1.0_rt-zd);
  weights[2] = xd*yd*(1.0_rt-zd);
  weights[3] = xd*(1.0_rt - yd)*(1.0_rt-zd);

  // zd = 2
  weights[4] = (1.0_rt - xd) *(1.0_rt - yd)*zd;
  weights[5] = (1.0_rt - xd) *yd*zd;
  weights[6] = xd*yd*zd;
  weights[7] = xd*(1.0_rt - yd)*zd;

  // TODO report if any of the weights are sld points

  // Real sum = 0.0;
  // Print() << "computeIPweights" << std::endl;
  // for (int kk=0; kk<8; kk++) {
  //   sum += weights[kk];
  //   Print() << kk+1 << " " << weights[kk] << std::endl;
  // }
  // Print() << "sum " << sum << std::endl;

  return weights;
}


void IB::computeGPs( int lev, MultiFab& consmf, MultiFab& primsmf, const PROB::ProbClosures& closures) {

  IBMultiFab& mfab = *ibMFa[lev];
  Vector<Array<Real,NPRIM>> stateIMs;
  Array<Real,NPRIM> stateIB,stateGP;
  stateIMs.resize(NUM_IMPS);
  int itemp, jtemp, ktemp;

  // for each fab in multifab (at a given level)
  for (MFIter mfi(mfab,false); mfi.isValid(); ++mfi) {

    // prims and cons fab
    auto const &conFab  = consmf.array(mfi); // this is a const becuase .array() returns a const but we can still modify conFab as consmf input argument is not const
    auto const &primFab = primsmf.array(mfi);

    IBM::IBFab& ibFab = mfab.get(mfi);


    // for each ghost point
    for (int ii=0;ii<ibFab.ngps;ii++) {
      IntArray& indexGP = ibFab.gpArray[ii].idx;

      // interpolate IM
      // for each image point
      for (int jj=0; jj<NUM_IMPS; jj++) {
        Array<int,AMREX_SPACEDIM>& indexIM = ibFab.gpArray[ii].impscell[jj]; 
        for (int kk=0; kk<NPRIM; kk++) {
          stateIMs[jj][kk] = 0.0_rt; // reset to 0
        }

        Vector<Array<Real,8>>& weightsIP = ibFab.gpArray[ii].ipweights;
        // for each IP point
        for (int ll=0; ll<8; ll++) {
          itemp = indexIM[0] + IB::indexCube[ll][0];
          jtemp = indexIM[1] + IB::indexCube[ll][1];
          ktemp = indexIM[2] + IB::indexCube[ll][2];
          // for each primitive variable
          for (int kk=0; kk<NPRIM; kk++) {
            stateIMs[jj][kk] += primFab(itemp,jtemp,ktemp,kk)*weightsIP[jj][ll];
            // Print() << kk <<   << " " << primFab(itemp,jtemp,ktemp,kk)  << std::endl;
          }
        }
          // for (int kk=0; kk<NPRIM; kk++) {
          //   Print() << kk << " " << stateIMs[jj][kk] << std::endl ;
          // }
      }


      // coordinate transformation for slip BCs --> need normals here!

      // interpolate IB (enforcing BCs) -- TODO:generalise to IBM BCs and make user defined
      // no-slip velocity
      stateIB[QU] = 0.0_rt;
      stateIB[QV] = 0.0_rt;
      stateIB[QW] = 0.0_rt;
      // zerograd temperature and pressure
      stateIB[QPRES] = stateIMs[0][QPRES];
      stateIB[QT]    = stateIMs[0][QT];
      // ensure thermodynamic consistency
      stateIB[QRHO]  = stateIMs[0][QPRES]/(stateIMs[0][QT]*closures.Rspec); 


      // extrapolate
      // linear extrapolation
      extrapolateGP( stateGP, stateIB, stateIMs,  ibFab.gpArray[ii].disGP, disIM[lev]);
      // thermodynamic consistency
      stateGP[QRHO]  = stateGP[QPRES]/(stateGP[QT]*closures.Rspec);

      // transfer GP to consfab and primfab
      itemp = indexGP[0];
      jtemp = indexGP[1];
      ktemp = indexGP[2];
      // insert Primitive ghost state into primFab 
      for (int n = 0; n < NPRIM; n++)
      {
        primFab(indexGP[0],indexGP[1],indexGP[2],n) = stateGP[n];
      }

      // insert conservative ghost state into consFab
      conFab(itemp,jtemp,ktemp,URHO) = stateGP[QRHO];
      conFab(itemp,jtemp,ktemp,UMX)  = stateGP[QRHO]*stateGP[QU];
      conFab(itemp,jtemp,ktemp,UMY)  = stateGP[QRHO]*stateGP[QV];
      conFab(itemp,jtemp,ktemp,UMZ)  = stateGP[QRHO]*stateGP[QW];
      Real ek   = 0.5_rt*(stateGP[QU]*stateGP[QU] + stateGP[QV]*stateGP[QV] + stateGP[QW]*stateGP[QW]);
      conFab(itemp,jtemp,ktemp,UET) = stateGP[QPRES]*(closures.gamma-1.0_rt) + stateGP[QRHO]*ek;

    }
  }
}

// linear extrapolation
void IB::extrapolateGP(Array<Real,6>& stateGP, Array<Real,6>& stateIB, Vector<Array<Real,6>>& stateIMs, Real dgp, Real dim) {

  for (int kk=0; kk<NPRIM; kk++) {
    stateGP[kk] = stateIB[kk] - (dgp/dim)*(stateIMs[0][kk] - stateIB[kk]);
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

  PMP::compute_face_normals(geom, boost::make_assoc_property_map(fnormals));
  Print() << "Face normals computed" << std::endl;

  // plane class also computes orthogonal direction to the face. However, the orthogonal vector is not normalised.
  std::for_each( geom.facets_begin(), geom.facets_end(), compute_plane_equations);
  
  // create face to displacement map //
  // auto temp = boost::make_assoc_property_map(fdisplace);
  // for(face_descriptor f : faces(geom))
  // {
  //   Vector_CGAL vec;
  //   put(temp, f, vec);
  //   // std::cout << "face plane " << f->plane() << "\n";
  // }

  // create face to surfdata map //
  auto map = boost::make_assoc_property_map(face2state);
  for(face_descriptor f : faces(geom))
  {
    surfdata data;
    put(map, f, data);
    // std::cout << "face plane" << f->plane() << "\n";
  }
}

void IB::moveGeom() {
  // Displace verticies //
  // For each vertex its position p is translated by the displacement vector (di) for each ith face. Each vertex has nfaces, for a triangular closed mesh this equals the number of edges at a vertex. This is called degree of the vertex by CGGAL.
  for (Polyhedron::Facet_handle fh : geom.facet_handles())
  {
    // Print() << "New face" << " \n";
    Polyhedron::Halfedge_handle start = fh->halfedge(), h = start;
    do {
      int nfaces = h->vertex()->degree();
      CGAL::Point_3 p = h->vertex()->point();
      // std::cout << "Vertex degree = " << nfaces  << "\n";
      // std::cout << "Vertex before = " << p << "\n";
      face_descriptor f = fh->halfedge()->face();
      Array<Real,AMREX_SPACEDIM> dis = face2state[f].displace; 
      
      CGAL::Vector_3<K2> di(dis[0]/nfaces,dis[1]/nfaces,dis[2]/nfaces);
      CGAL::Aff_transformation_3<K2> translate(CGAL::TRANSLATION,di);
      p = p.transform(translate);

      // std::cout << "Vertex after = " << p << "\n";

      h = h->next();
    } while(h!=start);
  }

  // apply boundary conditions //
  // using a map of boundary nodes?
  

  // Misc geometry things //
  // rebuild tree
  treePtr->rebuild(faces(geom).first,faces(geom).second,geom);
  Print() << "Tree rebuilt" << std::endl;

  CGAL::Polygon_mesh_processing::compute_face_normals(geom, boost::make_assoc_property_map(fnormals));
  Print() << "Face normals recomputed" << std::endl;

  // plane class also computes orthogonal direction to the face. However, the orthogonal vector is not normalised.
  std::for_each( geom.facets_begin(), geom.facets_end(),compute_plane_equations);
  Print() << "Plane equations recomputed" << std::endl;

  // Geometry fair?
}

void IB::computeSurf(int lev) {

  // for each level 



  exit(0);
}


