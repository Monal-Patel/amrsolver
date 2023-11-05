#include <IBM.h>
#include <AMReX_ParmParse.H>

using namespace amrex;
using ibm_t = IBM::ibm_t;

// constructor
ibm_t::ibm_t() {}

// destructor
ibm_t::~ibm_t () {
  // clear memory
  for (int ii=0; ii<NGEOM; ii++) {
    delete VtreePtr.at(ii);
    delete VInOutFunc.at(ii);
  };
}

// initialise IB
void ibm_t::init(Amr* pointer_amr, int nghost) {

  NGHOST_IB = nghost;
  pamr = pointer_amr ; // store pointer to main Amr class object's instance
  ref_ratio = pamr->refRatio();
  MAX_LEVEL = pamr->maxLevel();
  ibMFa.resize(MAX_LEVEL + 1);
  lsMFa.resize(MAX_LEVEL + 1);

  cellSizes.resize(MAX_LEVEL+1);
  disIM.resize(MAX_LEVEL+1);
  cellSizes[0] = pamr->Geom(0).CellSizeArray();
  for (int i=1;i<=MAX_LEVEL;i++) {
    for (int j=0;j<AMREX_SPACEDIM;j++) {
      cellSizes[i][j] = cellSizes[i-1][j]/ref_ratio[i-1][j];
    }
  }

  // compute distance between image points
  for (int i=0;i<=MAX_LEVEL;i++) {
    disIM[i] = PROB::distance_ip*sqrt(cellSizes[i][0]*cellSizes[i][0] 
    + cellSizes[i][1]*cellSizes[i][1] + cellSizes[i][2]*cellSizes[i][2]);
  }

  // read geometry
  readGeom();
}

// create IBMultiFabs at a level and store pointers to it
void ibm_t::buildMFs (const BoxArray& bxa, const DistributionMapping& dm, int lev) {
  ibMFa[lev] = new IBMultiFab(bxa,dm,NVAR_IB,NGHOST_IB);
  lsMFa[lev].define(bxa,dm,1,NGHOST_IB);
}

void ibm_t::destroyMFs (int lev) {
  if (!ibMFa.empty()) {
      delete ibMFa.at(lev);
  }
  if (!lsMFa.empty()) {
      lsMFa[lev].clear();
  }
}

 void ibm_t::computeMarkers (int lev) {


  IBMultiFab& mfab = *ibMFa[lev];
  int nhalo = mfab.nGrow(0); // assuming same number of ghost points in all directions
  GpuArray<Real,AMREX_SPACEDIM> prob_lo = pamr->Geom(lev).ProbLoArray();

  for (MFIter mfi(mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &ibFab = mfab.get(mfi);
    const Box& bx = mfi.tilebox();
    const IntVect& lo = bx.smallEnd();
    const IntVect& hi = bx.bigEnd();
    auto const& ibMarkers = mfab.array(mfi); // boolean array
    
    // compute sld markers (including at ghost points) - cannot use ParallelFor - CGAL call causes problems
    for (int k = lo[2]-nhalo; k <= hi[2]+nhalo; ++k) {
    for (int j = lo[1]-nhalo; j <= hi[1]+nhalo; ++j) {
    for (int i = lo[0]-nhalo; i <= hi[0]+nhalo; ++i) {
      ibMarkers(i,j,k,0) = false; // initialise to false
      ibMarkers(i,j,k,1) = false; // initialise to false

      Real x=prob_lo[0] + (0.5_rt + Real(i))*cellSizes[lev][0];
      Real y=prob_lo[1] + (0.5_rt + Real(j))*cellSizes[lev][1];
      Real z=prob_lo[2] + (0.5_rt + Real(k))*cellSizes[lev][2];
      Point gridpoint(x,y,z);

      for (int ii=0; ii<ibm_t::NGEOM; ii++) {
        // Print() << i << " " << j << " " << k << " " << ii << std::endl;
        inside_t& inside = *VInOutFunc[ii];
        CGAL::Bounded_side result = inside(gridpoint);
        AMREX_ASSERT_WITH_MESSAGE((result != CGAL::ON_BOUNDARY),"Grid point on IB surface");
        // if point inside any IB geometry, mark as solid, move on to another point. This minimises the number of inout testing (expensive) calls.
        if (int(result) == int(CGAL::ON_BOUNDED_SIDE)) {
          ibMarkers(i,j,k,0) = true;
          break;}
      }

    }}};

    // compute ghost markers
    // TODO: move this to initialiseGPs
    ibFab.gpData.ngps=0;
    int nextra=1;
    for (int k = lo[2] - nextra; k <= hi[2] + nextra; ++k) {
    for (int j = lo[1] - nextra; j <= hi[1] + nextra; ++j) {
    for (int i = lo[0] - nextra; i <= hi[0] + nextra; ++i) {
      bool ghost = false;
      if (ibMarkers(i,j,k,0)) {
        for (int l = -1; l<=1; l=l+2) {
          ghost = ghost || (!ibMarkers(i+l,j,k,0));
          ghost = ghost || (!ibMarkers(i,j+l,k,0));
          ghost = ghost || (!ibMarkers(i,j,k+l,0));
        }
        ibMarkers(i,j,k,1) = ghost; 
        ibFab.gpData.ngps += ghost;

        if (ghost) {
          // store GP index
          ibFab.gpData.gp_ijk.push_back(Array1D<int,0,AMREX_SPACEDIM-1>{i,j,k});
      }
      else {
        ibMarkers(i,j,k,1) = false;
      }
    }}};
  }
  }
}

void ibm_t::initialiseGPs (int lev) {

  IBMultiFab& mfab = *ibMFa[lev];
  int nhalo = mfab.nGrow(0); // assuming same number of ghost points in all directions
  GpuArray<Real,AMREX_SPACEDIM> prob_lo = pamr->Geom(lev).ProbLoArray();

  for (MFIter mfi(mfab,false); mfi.isValid(); ++mfi) {
    IBM::IBFab &ibFab = mfab.get(mfi);
    gpData_t& gpData = ibFab.gpData;
    const Box& bxg = mfi.growntilebox(nhalo);
    // const Box& bx = mfi.tilebox();
    const IntVect& lo = bxg.smallEnd();
    const IntVect& hi = bxg.bigEnd();
    auto const& ibMarkers = mfab.array(mfi); // boolean array

    // we need a CPU loop here (cannot be GPU loop) as CGAL tree seach for closest element to a point needs to be called.
    // instead of looping through previously indexed gps, we loop through the whole ghost point field as it is available on GPU and CPU at all times. Unlike the gp indexes, which are only stored on GPU memory.
    // Array1D<int,0,AMREX_SPACEDIM-1>& idx = ibFab.gpData.gp_ijk[ii];

    for (int k = lo[2]; k <= hi[2]; ++k) {
    for (int j = lo[1]; j <= hi[1]; ++j) {
    for (int i = lo[0]; i <= hi[0]; ++i) {

    // for each ghost point
    if (ibMarkers(i,j,k,1)) {

      Real x=prob_lo[0] + (0.5_rt + i)*cellSizes[lev][0];
      Real y=prob_lo[1] + (0.5_rt + j)*cellSizes[lev][1];
      Real z=prob_lo[2] + (0.5_rt + k)*cellSizes[lev][2];
      Point gp(x,y,z);

      // find and store geometery index for this GP. This index is used for searching appropriate tree in initialiseGPs to find the closest element and searching all trees.

      // in out test for each geometry
      int igeom;
      for (int ii=0; ii<ibm_t::NGEOM; ii++) {
        inside_t& inside = *VInOutFunc[ii];
        CGAL::Bounded_side result = inside(gp);

        if (int(result) == int(CGAL::ON_BOUNDED_SIDE)) {
          igeom = ii;
          ibFab.gpData.geomIdx.push_back(igeom);
          break; // do not search other geometries if current point is found to be inside a geometry
        }
      // TODO: assert geometries do not overlap?
      }

      // closest surface point and face --------------------------
      Point_and_primitive_id closest_elem = VtreePtr[igeom]->closest_point_and_primitive(gp);

      //store
      gpData.closest_cgal.push_back(closest_elem);

      // This closest point (cp) is between the face plane and the gp
      Point cp = closest_elem.first;
      Polyhedron::Face_handle face = closest_elem.second; 

      // amrex::Print() << "------------------- " << std::endl;
      // Print() << "closest surface point: " << cp << std::endl;
      // Print() << "closest triangle: ( "
      //           << face->halfedge()->vertex()->point() << " , "
      //           << face->halfedge()->next()->vertex()->point() << " , "
      //           << face->halfedge()->next()->next()->vertex()->point()
      //           << " )" 
      //           << std::endl; 
      // Print() << "Normal " << Vfnormals[igeom][face] <<std::endl;
      // Print() << "cp-gp " << cp - gp << std::endl; // should be in the direction of normal
      // Print() << "Plane " << face->plane().a() << " " << face->plane().b() << " "  << face->plane().c() << " " << face->plane().d() << std::endl;

      // IB point -------------------------------------------
      Vector_CGAL imp_gp(gp,cp);
      Real disGP = sqrt(CGAL::squared_distance(gp,cp));
      AMREX_ASSERT_WITH_MESSAGE(disGP < 1.0*sqrt(cellSizes[lev][0]*cellSizes[lev][0] + cellSizes[lev][1]*cellSizes[lev][1] + cellSizes[lev][2]*cellSizes[lev][2]), "Ghost point and IB point distance larger than mesh diagonal");


      //*store*
      gpData.disGP.push_back(disGP);
      Array1D<Real,0,AMREX_SPACEDIM-1> norm = {Vfnormals[igeom][face][0],Vfnormals[igeom][face][1],Vfnormals[igeom][face][2]};

      Point p1 = face->halfedge()->vertex()->point();
      Point p2 = face->halfedge()->next()->vertex()->point();
      Vector_CGAL tan1_not_unit(p1,p2); 
      Real len = sqrt(CGAL::squared_distance(p1,p2));
      Array1D<Real,0,AMREX_SPACEDIM-1> tan1;
      tan1(0) = tan1_not_unit[0]/len; tan1(1) = tan1_not_unit[1]/len; tan1(2) = tan1_not_unit[2]/len;

      // norm x tan1
      Array1D<Real,0,AMREX_SPACEDIM-1> tan2 = {norm(1)*tan1(2)-norm(2)*tan1(1), norm(2)*tan1(0)-norm(0)*tan1(2), norm(0)*tan1(1)-norm(1)*tan1(0)};
      
      // norm.tan1 and norm.tan2 == 0
      AMREX_ASSERT_WITH_MESSAGE( (norm(0)*tan1(0) + norm(1)*tan1(1) + norm(2)*tan1(2) + norm(0)*tan2(0) + norm(1)*tan2(1) + norm(2)*tan2(2)) < 1.0e-9,"norm.tan1 or norm.tan2 not orthogonal");

      gpData.normal.push_back(norm);
      gpData.tangent1.push_back(tan1);
      gpData.tangent2.push_back(tan2);

      // IM points -------------------------------------------
      Array2D<Real,0,NIMPS-1,0,AMREX_SPACEDIM-1> imp_xyz; 
      Array2D<int,0,NIMPS-1,0,AMREX_SPACEDIM-1> imp_ijk; 

      // find image point and the bottom left point closest to the image point
      //  In 2D, same idea in 3D.
      //     i,j+1 (2) ---------------------     i+1,j+1 (3)
      //     |                                  |
      //     |         P                        |
      //     |                                  |
      //     |                                  |
      //     i,j  (1) ----------------------      i+1,j  (4)
      for (int jj=0; jj<NIMPS; jj++) {
        for (int kk=0; kk<AMREX_SPACEDIM; kk++) {
          imp_xyz(jj,kk) = cp[kk] + Real(jj+1)*disIM[lev]*Vfnormals[igeom][face][kk];
          imp_ijk(jj,kk) = floor((imp_xyz(jj,kk) - prob_lo[kk])/cellSizes[lev][kk] - 0.5_rt);
        }
 
        AMREX_ASSERT_WITH_MESSAGE(bxg.contains(imp_ijk(jj,0),imp_ijk(jj,1),imp_ijk(jj,2)),"Interpolation point outside fab");
      }
      // DEBUGGING //////////////
      // Print() << "disGP, disIM (from IB) " << fab.gpArray[ii].disGP << " " << ibm_t::disIM[lev] << std::endl;
      // for (int jj=0; jj<NIMPS; jj++) {
      //   Print() << "imp" << jj+1 << " "<< fab.gpArray[ii].imps[jj] << std::endl;
      //   Print() << "impCell" << jj+1 << " "<< fab.gpArray[ii].impscell[jj] << std::endl;
      // }
      ///////////////////////////
      // *store*
      gpData.disIM.push_back(disIM[lev]); 
      gpData.imp_xyz.push_back(imp_xyz);
      gpData.imp_ijk.push_back(imp_ijk);

      // Interpolation points' (ips) weights for each image point
      Array2D<Real,0,NIMPS-1,0,7> ipweights;
      Array3D<int,0,NIMPS-1,0,7,0,AMREX_SPACEDIM-1> ip_ijk;
      computeIPweights(ipweights,ip_ijk,imp_xyz, imp_ijk, prob_lo, cellSizes[lev], ibMarkers);
      // *store*
      gpData.imp_ipweights.push_back(ipweights);
      gpData.imp_ip_ijk.push_back(ip_ijk);

      //  if (k==35) {
        // int jj = 0;
        // Print() << "--- " << std::endl;
      //   Print() << "gp array idx " << gpData.normal.size() - 1 << std::endl;
      //   Print() << "bxg " << bxg << std::endl;
      //   Print() << "im_ijk " << imp_ijk(jj,0) << " " << imp_ijk(jj,1) << " " << imp_ijk(jj,2) << std::endl;
      //   Print() << "gp_xyz " << x << " " << y << " " << z << std::endl;
      //   Print() << "ib_xyz " << cp[0] << " " << cp[1] << " " << cp[2] << std::endl;
      //   Print() << "ip_xyz " << imp_xyz(jj,0) << " " << imp_xyz(jj,1) << " " << imp_xyz(jj,2) << std::endl;
      //   Print() << "norm " << norm(0) << " " << norm(1) << " " << norm(2) << std::endl;
      //   Print() << "tan1 " << tan1(0) << " " << tan1(1) << " " << tan1(2) << std::endl;
      //   Print() << "tan2 " << tan2(0) << " " << tan2(1) << " " << tan2(2) << std::endl;
      //   Print() << "dx  " << cellSizes[lev][0] << " " << cellSizes[lev][1] << " " << cellSizes[lev][2] << std::endl;
      //   Real temp = sqrt(cellSizes[lev][0]*cellSizes[lev][0] + cellSizes[lev][1]*cellSizes[lev][1] + cellSizes[lev][2]*cellSizes[lev][2]);
      //   Print() << "dx diag " << temp << std::endl;
      //   Print() << "disIM "  << disIM[lev] << std::endl;
      //   Print() << "disGP " << disGP << std::endl;
      // Print() << "gp_ijk " << i << " " << j << " " << k << std::endl;
      // for (int iip=0; iip<8; iip++) {
      //   Print() << "ip " << iip << "  ijk " << ip_ijk(jj,iip,0) << " " << ip_ijk(jj,iip,1) << " " << ip_ijk(jj,iip,2) << "      weight " << " " << ipweights(jj,iip) << std::endl;
      // }

      }
    }
    }
    }
  }
}


void ibm_t::computeGPs( int lev, MultiFab& consmf, MultiFab& primsmf, IBMultiFab& ibmf, const closures& cls) {

  // for each fab in multifab (at a given level)
  for (MFIter mfi(ibmf,false); mfi.isValid(); ++mfi) {
    
    // for GP data
    auto& ibFab = ibmf.get(mfi);

    // field arrays
    auto const& cons  = consmf.array(mfi); // this is a const becuase .array() returns a const but we can still modify cons as consmf input argument is not const
    auto const& prims = primsmf.array(mfi);
    auto const& markers = ibmf.array(mfi);

    auto const gp_ijk = ibFab.gpData.gp_ijk.data();
    auto const imp_ijk= ibFab.gpData.imp_ijk.data();
    auto const imp_ipweights = ibFab.gpData.imp_ipweights.data();
    auto const imp_ip_ijk = ibFab.gpData.imp_ip_ijk.data();
    // TODO: combine disGP and disIM into 2D array
    auto const disGP = ibFab.gpData.disGP.data();
    auto const disIM = ibFab.gpData.disIM.data();
    // TODO: combine norm,tan1,tan2 into matrix (2d array)
    auto const norm = ibFab.gpData.normal.data();
    auto const tan1 = ibFab.gpData.tangent1.data();
    auto const tan2 = ibFab.gpData.tangent2.data();

    ParallelFor(ibFab.gpData.ngps, [=,copy=this] AMREX_GPU_DEVICE (int ii)
    {

      Array2D<Real,0,NIMPS+1,0,NPRIM-1> primsNormal={0.0};
      copy->interpolateIMs(imp_ip_ijk[ii],imp_ipweights[ii],prims,primsNormal);

      // transform velocity to local coordinates for image points only
      for (int iip=2; iip<2+NIMPS; iip++) {
        copy->global2local(iip, primsNormal, norm[ii], tan1[ii], tan2[ii]);
      }
      copy->computeIB(primsNormal,cls);
      copy->extrapolate(primsNormal, disGP[ii], disIM[ii]);
      // thermodynamic consistency
      primsNormal(0,QRHO)  = primsNormal(0,QPRES)/(primsNormal(0,QT)*cls.Rspec);

      // only transform velocity back to global coordinates for gp only
      int idx=0;
      copy->local2global(idx,primsNormal,norm[ii],tan1[ii],tan2[ii]);

      // limiting p and T
      // primsNormal(0,QPRES) = max(primsNormal(0,QPRES),1.0);
      // primsNormal(0,QT)    = max(primsNormal(0,QT),50.0);

      // insert primitive variables into primsFab
      int i=gp_ijk[ii](0); int j=gp_ijk[ii](1); int k = gp_ijk[ii](2);
      for (int nn=0; nn<NPRIM; nn++) {
        prims(i,j,k,nn) = primsNormal(0,nn);
      }

      // AMREX_ASSERT_WITH_MESSAGE( prims(i,j,k,QPRES)>50,"P<50 at GP");

      // insert conservative ghost state into consFab
      cons(i,j,k,URHO) = primsNormal(0,QRHO);
      cons(i,j,k,UMX)  = primsNormal(0,QRHO)*primsNormal(0,QU);
      cons(i,j,k,UMY)  = primsNormal(0,QRHO)*primsNormal(0,QV);
      cons(i,j,k,UMZ)  = primsNormal(0,QRHO)*primsNormal(0,QW);
      Real ek   = 0.5_rt*(primsNormal(0,QU)*primsNormal(0,QU) + primsNormal(0,QV)* primsNormal(0,QV) + primsNormal(0,QW)*primsNormal(0,QW));
      cons(i,j,k,UET) = primsNormal(0,QPRES)/(cls.gamma-1.0_rt) + primsNormal(0,QRHO)*ek;
      });
    }
  }

void ibm_t::compute_plane_equations( Polyhedron::Facet& f) {
    Polyhedron::Halfedge_handle h = f.halfedge();
    f.plane() = Polyhedron::Plane_3( h->opposite()->vertex()->point(), 
		       h->vertex()->point(),
		       h->next()->vertex()->point());
};

void ibm_t::readGeom() {

  ParmParse pp;
  Vector<std::string> Vfilename;
  pp.getarr("ib.filename",Vfilename);

  ibm_t::NGEOM = Vfilename.size();
  VGeom.resize(NGEOM);
  VtreePtr.resize(NGEOM);
  Vfnormals.resize(NGEOM);
  VInOutFunc.resize(NGEOM);

  namespace PMP = CGAL::Polygon_mesh_processing;
  Print() << "----------------------------------" << std::endl;
  for (int i=0; i<NGEOM; i++) {
    Print() << "----------------------------------" << std::endl;
    if(!PMP::IO::read_polygon_mesh(Vfilename[i], ibm_t::VGeom[i]))
    {
      std::cerr << "Invalid geometry filename" << std::endl;
      exit(1);
    }
    Print() << "Geometry (i=" << i << ") " << Vfilename[i] << " read"<< std::endl;
    Print() << "Is geometry only made of triangles? " << VGeom[i].is_pure_triangle() << std::endl;
    Print() << "Number of facets " << VGeom[i].size_of_facets() << std::endl;

  // constructs AABB tree and computes internal KD-tree
  // data structure to accelerate distance queries
    VtreePtr[i] = new Tree (faces(VGeom[i]).first, faces(VGeom[i]).second, VGeom[i]);
    Print() << "AABB tree constructed" << std::endl;

    PMP::compute_face_normals(VGeom[i], boost::make_assoc_property_map(Vfnormals[i]));
    Print() << "Face normals computed" << std::endl;

    // plane class also computes orthogonal direction to the face. However, the orthogonal vector is not normalised.
    std::for_each( VGeom[i].facets_begin(), VGeom[i].facets_end(), compute_plane_equations);
    Print() << "Plane equations per face computed" << std::endl;

    // make inside/outside function for each geometry
    for (int ii=0; ii<ibm_t::NGEOM; ii++) {
      VInOutFunc[ii] = new inside_t(ibm_t::VGeom[ii]);
    }
    Print() << "In out testing function constructed" << std::endl;

    // create face to displacement map //
    // auto temp = boost::make_assoc_property_map(fdisplace);
    // for(face_descriptor f : faces(geom))
    // {
    //   Vector_CGAL vec;
    //   put(temp, f, vec);
    //   // std::cout << "face plane " << f->plane() << "\n";
    // }

    // create face to surfdata map //
    // auto map = boost::make_assoc_property_map(face2state);
    // for(face_descriptor f : faces(geom))
    // {
    //   surfdata data;
    //   put(map, f, data);
    //   // std::cout << "face plane" << f->plane() << "\n";
    // }
  }
  Print() << "----------------------------------" << std::endl;
  Print() << "----------------------------------" << std::endl;

  //  || CGAL::is_empty(ibm_t::VGeom[i]) || !CGAL::is_triangle_mesh(ibm_t::geom)

}

// void ibm_t::moveGeom() {
//   // Displace verticies //
//   // For each vertex its position p is translated by the displacement vector (di) for each ith face. Each vertex has nfaces, for a triangular closed mesh this equals the number of edges at a vertex. This is called degree of the vertex by CGGAL.
//   for (Polyhedron::Facet_handle fh : geom.facet_handles())
//   {
//     // Print() << "New face" << " \n";
//     Polyhedron::Halfedge_handle start = fh->halfedge(), h = start;
//     do {
//       int nfaces = h->vertex()->degree();
//       CGAL::Point_3 p = h->vertex()->point();
//       // std::cout << "Vertex degree = " << nfaces  << "\n";
//       // std::cout << "Vertex before = " << p << "\n";
//       face_descriptor f = fh->halfedge()->face();
//       Array<Real,AMREX_SPACEDIM> dis = face2state[f].displace; 
      
//       CGAL::Vector_3<K2> di(dis[0]/nfaces,dis[1]/nfaces,dis[2]/nfaces);
//       CGAL::Aff_transformation_3<K2> translate(CGAL::TRANSLATION,di);
//       p = p.transform(translate);

//       // std::cout << "Vertex after = " << p << "\n";

//       h = h->next();
//     } while(h!=start);
//   }

//   // apply boundary conditions //
//   // using a map of boundary nodes?
  

//   // Misc geometry things //
//   // rebuild tree
//   treePtr->rebuild(faces(geom).first,faces(geom).second,geom);
//   Print() << "Tree rebuilt" << std::endl;

//   CGAL::Polygon_mesh_processing::compute_face_normals(geom, boost::make_assoc_property_map(fnormals));
//   Print() << "Face normals recomputed" << std::endl;

//   // plane class also computes orthogonal direction to the face. However, the orthogonal vector is not normalised.
//   std::for_each( geom.facets_begin(), geom.facets_end(),compute_plane_equations);
//   Print() << "Plane equations recomputed" << std::endl;

//   // Geometry fair?
// }

void ibm_t::computeSurf(int lev) {

  // for each level 



  exit(0);
}






