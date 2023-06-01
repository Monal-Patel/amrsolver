#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <CNS.H>
#include <CNS_K.H>
#include <cns_prob.H>
#include <CNS_parm.H>

#ifdef AMREX_USE_GPIBM
#include <IBM.H>
#endif

using namespace amrex;

int CNS::NGHOST;

bool CNS::rhs_euler=false;
bool CNS::rhs_visc=false;
bool CNS::rhs_source=false;
bool CNS::verbose = true;
bool CNS::dt_dynamic=false;
int  CNS::nstep_screen_output=10;
int  CNS::flux_euler=0;
int  CNS::order_keep=2;
int  CNS::order_rk=2;
int  CNS::do_reflux = 1;
int  CNS::refine_max_dengrad_lev = -1;
Real CNS::cfl = 0.0;
Real CNS::dt_constant = 0.0;
Real CNS::refine_dengrad = 1.0e10;

Real CNS::gravity = 0.0;

// needed for CNSBld - derived from LevelBld (abstract class, pure virtual functions must be implemented)

CNS::CNS() {}

CNS::CNS(Amr &papa,
         int lev,
         const Geometry &level_geom,
         const BoxArray &bl,
         const DistributionMapping &dm,
         Real time)
    : AmrLevel(papa, lev, level_geom, bl, dm, time)
{
  if (do_reflux && level > 0)
  {
    flux_reg.reset(new FluxRegister(grids, dmap, crse_ratio, level, NCONS));
  }

  buildMetrics();
}

CNS::~CNS() {}
// -----------------------------------------------------------------------------

// init ------------------------------------------------------------------------

void CNS::read_params()
{
  ParmParse pp("cns");

  pp.query("verbose", verbose);

  Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
  pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
  pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
  for (int i = 0; i < AMREX_SPACEDIM; ++i)
  {
    h_phys_bc->setLo(i, lo_bc[i]);
    h_phys_bc->setHi(i, hi_bc[i]);
  }

  pp.query("rhs_euler", rhs_euler);
  pp.query("rhs_visc" , rhs_visc);
  pp.query("rhs_source",rhs_source);
  pp.query("do_reflux", do_reflux);

  if (!pp.query("flux_euler",flux_euler)) {
    amrex::Abort("Need to specify Euler flux type.");}
  
  if (flux_euler==1) {
    if (!pp.query("order_keep",order_keep)) {
      amrex::Abort("Need to specify KEEP scheme order of accuracy (2, 4 or 6)");
    }
  }

  if (!pp.query("order_rk",order_rk)) {
    amrex::Abort("Need to specify SSPRK scheme order of accuracy (2 or 3)");
  }

  pp.query("refine_max_dengrad_lev", refine_max_dengrad_lev);
  pp.query("refine_dengrad", refine_dengrad);

  pp.query("gravity", gravity);

#if AMREX_USE_GPU
  amrex::Gpu::htod_memcpy(d_parm, h_parm, sizeof(Parm));
  amrex::Gpu::htod_memcpy(d_prob_parm, h_prob_parm, sizeof(ProbParm));
  amrex::Gpu::htod_memcpy(d_phys_bc, h_phys_bc, sizeof(BCRec));
#endif
}

void CNS::init(AmrLevel &old)
{
  auto &oldlev = dynamic_cast<CNS &>(old);

  Real dt_new = parent->dtLevel(level);
  Real cur_time = oldlev.state[State_Type].curTime();
  Real prev_time = oldlev.state[State_Type].prevTime();
  Real dt_old = cur_time - prev_time;
  setTimeLevel(cur_time, dt_old, dt_new);

  MultiFab &S_new = get_new_data(State_Type);
  FillPatch(old, S_new, 0, cur_time, State_Type, 0, NCONS);
}

void CNS::init()
{
  Real dt = parent->dtLevel(level);
  Real cur_time = getLevel(level - 1).state[State_Type].curTime();
  Real prev_time = getLevel(level - 1).state[State_Type].prevTime();
  Real dt_old = (cur_time - prev_time) / static_cast<Real>(parent->MaxRefRatio(level - 1));
  setTimeLevel(cur_time, dt_old, dt);

  MultiFab &S_new = get_new_data(State_Type);
  FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NCONS);
};

void CNS::initData()
{
  BL_PROFILE("CNS::initData()");

  const auto geomdata = geom.data();
  MultiFab &S_new = get_new_data(State_Type);
  // S_new = 0.0; // default initialistiaon with Nans (preferred).

  Parm const *lparm = d_parm;
  ProbParm const *lprobparm = d_prob_parm;

  auto const &sma = S_new.arrays();
  amrex::ParallelFor(S_new,
                     [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
                     {
                       user_initdata(i, j, k, sma[box_no], geomdata, *lparm, *lprobparm);
                     });

  // Compute the initial temperature (will override what was set in initdata)
  // computeTemp(S_new, 0);

  // MultiFab& C_new = get_new_data(Cost_Type);
  // C_new.setVal(1.0);
}

void CNS::buildMetrics()
{
  // print mesh sizes
  const Real *dx = geom.CellSize();
  amrex::Print() << "Mesh size (dx,dy,dz) = ";
  amrex::Print() << dx[0] << "  "
                 << dx[1] << "  "
                 << dx[2] << "  \n";
}

void CNS::post_init(Real)
{
  if (level > 0) {return;};

  for (int k = parent->finestLevel() - 1; k >= 0; --k)
  {
    getLevel(k).avgDown();
  }

  if (verbose) {
  printTotal();
  }
}
// -----------------------------------------------------------------------------

// Time-stepping ---------------------------------------------------------------
void CNS::computeTemp(MultiFab &State, int ng)
{
  BL_PROFILE("CNS::computeTemp()");

  Parm const *lparm = d_parm;

  // This will reset Eint and compute Temperature
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box &bx = mfi.growntilebox(ng);
    auto const &sfab = State.array(mfi);

    amrex::Abort("ComputeTemp function not written");

    amrex::ParallelFor(bx,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                       {
                         cns_compute_temperature(i, j, k, sfab, *lparm);
                       });
  }
}
// Called on level 0 only, after field data is initialised but before first step. 
void CNS::computeInitialDt(int finest_level,
                           int sub_cycle,
                           Vector<int> &n_cycle, // no. of subcycling steps
                           const Vector<IntVect> &ref_ratio,
                           Vector<Real> &dt_level,
                           Real stop_time)
{
  Real dt0=std::numeric_limits<Real>::max();
  int nfactor = 1;

  // dt at base level
  if (!dt_dynamic) {dt0 = dt_constant;}
  else {
    // estimate dt per level
    for (int i = 0; i <= finest_level; i++) {
      dt_level[i] = getLevel(i).estTimeStep(); }
    // find min dt across all levels
    nfactor=1;
    for (int i = 0; i <= finest_level; i++) {
        nfactor *= n_cycle[i];
        dt0 = std::min(dt0,nfactor*dt_level[i]); }
  }

  // dt at all levels
  nfactor = 1;
  for (int i = 0; i <= finest_level; i++)
    {
        nfactor *= n_cycle[i];
        dt_level[i] = dt0/nfactor;
    }

}


// Called at the end of a coarse grid timecycle or after regrid, to compute the dt (time step) for all levels, for the next step.
void CNS::computeNewDt(int finest_level,
                       int /*sub_cycle*/,
                       Vector<int> &n_cycle,
                       const Vector<IntVect> & /*ref_ratio*/,
                       Vector<Real> &dt_min,
                       Vector<Real> &dt_level,
                       Real stop_time,
                       int post_regrid_flag)
{
  // dt is constant, dt at level 0 is from inputs and dt at higher levels is computed from the number of subcycles ////////////////////////////////////////
  if(!dt_dynamic)   {
    int nfactor = 1;
    for (int i = 0; i <= finest_level; i++) {
      nfactor *= n_cycle[i];
      dt_level[i] = dt_constant / nfactor;}
    return;
  }

  // if dt is dynamic //////////////////////////////////////////////////////////
  for (int i = 0; i <= finest_level; i++) {
    dt_min[i] = getLevel(i).estTimeStep();
  }

  // Limit dt
  if (post_regrid_flag == 1) {
    // Limit dt's by pre-regrid dt
    for (int i = 0; i <= finest_level; i++) {
      dt_min[i] = std::min(dt_min[i],dt_level[i]);}
  }
  // Limit dt's by change_max * old dt
  else {
    static Real change_max = 1.1;
    for (int i = 0; i <= finest_level; i++)
    { dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);}
  }

  // Find the minimum over all levels
  Real dt_0 = std::numeric_limits<Real>::max();
  int nfactor = 1;
  for (int i = 0; i <= finest_level; i++) {
    nfactor *= n_cycle[i];
    dt_0 = std::min(dt_0,nfactor*dt_min[i]);}

  // Limit dt's by the value of stop_time.
  const Real eps = 0.001*dt_0;
  Real cur_time  = state[State_Type].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps)) {
        dt_0 = stop_time - cur_time;}
  }

  nfactor = 1;
  for (int i = 0; i <= finest_level; i++) {
    nfactor *= n_cycle[i];
    dt_level[i] = dt_0/nfactor;}
}


Real CNS::estTimeStep () {
  BL_PROFILE("CNS::estTimeStep()");

  const auto dx = geom.CellSizeArray();
  const MultiFab& S = get_new_data(State_Type);
  Parm const* lparm = d_parm;

  Real estdt = amrex::ReduceMin(S, 0,
  [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) -> Real { 
    return cns_estdt(bx, fab, dx, *lparm); });

  estdt *= cfl;
  ParallelDescriptor::ReduceRealMin(estdt);

  return estdt;
}


void CNS::post_timestep(int /* iteration*/)
{
  BL_PROFILE("post_timestep");

  if (do_reflux && level < parent->finestLevel()){
    MultiFab &S = get_new_data(State_Type);
    CNS &fine_level = getLevel(level + 1);
    fine_level.flux_reg->Reflux(S, Real(1.0), 0, 0, NCONS, geom);
  }

  if (level < parent->finestLevel()) { avgDown();}

  if (verbose && this->nStep()%nstep_screen_output == 0) {
    printTotal();}
}
// -----------------------------------------------------------------------------

// Gridding -------------------------------------------------------------------
void CNS::post_regrid(int lbase, int new_finest)
{

#ifdef AMREX_USE_GPIBM
  IBM::ib.destroyIBMultiFab(level);
  IBM::ib.buildIBMultiFab(this->boxArray(), this->DistributionMap(), level, 2, 2);
  IBM::ib.computeMarkers(level);
  IBM::ib.initialiseGPs(level);
#endif
}

void CNS::errorEst(TagBoxArray &tags, int /*clearval*/, int /*tagval*/,
                   Real time, int /*n_error_buf*/, int /*ngrow*/)
{
  // MF without ghost points filled (why?)
  MultiFab sdata(get_new_data(State_Type).boxArray(), get_new_data(State_Type).DistributionMap(), NCONS, 1, MFInfo(), Factory());

  // filling ghost points (copied from PeleC)
  const Real cur_time = state[State_Type].curTime();
  FillPatch(*this, sdata, sdata.nGrow(), cur_time, State_Type, 0, NCONS, 0);

#ifdef AMREX_USE_GPIBM
  // call function from cns_prob
  IBM::IBMultiFab *ibdata = IBM::ib.mfa[level];
  user_tagging(tags, sdata, level, ibdata);
#else
  user_tagging(tags, sdata, level);
#endif

}
// -----------------------------------------------------------------------------

// misc ------------------------------------------------------------------------

// const IntVect& len = bx.length();

void CNS::ghostboxes(int ng, const Box& bx, Array<Box,AMREX_SPACEDIM*2>& boxes){

  int imin  = bx.smallEnd(0);
  int imax  = bx.bigEnd(0);
  int igmax = imax + ng;
  int igmin = imin - ng;

  int jmin  = bx.smallEnd(1);
  int jmax  = bx.bigEnd(1);
  int jgmax = jmax + ng;
  int jgmin = jmin - ng;

  int kmin  = bx.smallEnd(2);
  int kmax  = bx.bigEnd(2);
  int kgmax = kmax + ng;
  int kgmin = kmin - ng;

  IntVect big,small;

  // left box
  big[0]=imin-1; big[1]=jmax; big[2]=kmax;
  small[0]=igmin; small[1]=jmin; small[2]=kmin;
  boxes[0].setBig(big); boxes[0].setSmall(small);

  // right box
  big[0]=igmax; big[1]=jmax; big[2]=kmax;
  small[0]=imax+1; small[1]=jmin; small[2]=kmin;
  boxes[1].setBig(big); boxes[1].setSmall(small);

  // top box
  big[0]=imax; big[1]=jgmax; big[2]=kmax;
  small[0]=imin; small[1]=jmax+1; small[2]=kmin;
  boxes[2].setBig(big); boxes[2].setSmall(small);

  // bottom box
  big[0]=imax; big[1]=jmin-1; big[2]=kmax;
  small[0]=imin; small[1]=jgmin; small[2]=kmin;
  boxes[3].setBig(big); boxes[3].setSmall(small);

  // front box
  big[0]=imax; big[1]=jmax; big[2]=kmin-1;
  small[0]=imin; small[1]=jmin; small[2]=kgmin;
  boxes[4].setBig(big); boxes[4].setSmall(small);

  // back box
  big[0]=imax; big[1]=jmax; big[2]=kgmax;
  small[0]=imin; small[1]=jmin; small[2]=kmax+1;
  boxes[5].setBig(big); boxes[5].setSmall(small);
}


void CNS::avgDown()
{
  BL_PROFILE("CNS::avgDown()");

  if (level == parent->finestLevel())
    return;

  auto &fine_lev = getLevel(level + 1);

  MultiFab &S_crse = get_new_data(State_Type);
  MultiFab &S_fine = fine_lev.get_new_data(State_Type);

  amrex::average_down(S_fine, S_crse, fine_lev.geom, geom,
                      0, S_fine.nComp(), parent->refRatio(level));

  // const int nghost = 0;
  // computeTemp(S_crse, nghost);
}

void CNS::printTotal() const
{
  const MultiFab &S_new = get_new_data(State_Type);
  std::array<Real, 5> tot;
  for (int comp = 0; comp < 5; ++comp)
  {
    tot[comp] = S_new.sum(comp, true) * geom.ProbSize();
  }
#ifdef BL_LAZY
  Lazy::QueueReduction([=]() mutable {
#endif
  ParallelDescriptor::ReduceRealSum(tot.data(), 5, ParallelDescriptor::IOProcessorNumber());

  amrex::Print().SetPrecision(17) 
  <<"\n[CNS level "<< level << "]\n" 
  <<"   Total mass   = " << tot[0] << "\n"
  <<"   Total x-mom  = " << tot[1] << "\n"
  <<"   Total y-mom  = " << tot[2] << "\n"
  <<"   Total z-mom  = " << tot[3] << "\n"
  <<"   Total energy = " << tot[4] << "\n";

#ifdef BL_LAZY
                       });
#endif
}

// Plotting
//------------------------------------------------------------------------------
void CNS::writePlotFile(const std::string& dir, std::ostream& os, VisMF::How how) {
  int i, n;
  //
  // The list of indices of State to write to plotfile.
  // first component of pair is state_type,
  // second component of pair is component # within the state_type
  //
  std::vector<std::pair<int, int>> plot_var_map;
  for (int typ = 0; typ < desc_lst.size(); typ++)
  {
    for (int comp = 0; comp < desc_lst[typ].nComp(); comp++)
    {
      if (parent->isStatePlotVar(desc_lst[typ].name(comp)) &&
          desc_lst[typ].getType() == IndexType::TheCellType())
      {
        plot_var_map.push_back(std::pair<int, int>(typ, comp));
      }
    }
  }

  int num_derive = 0;
  std::vector<std::string> derive_names;
  const std::list<DeriveRec> &dlist = derive_lst.dlist();
  for (auto const &d : dlist)
  {
    if (parent->isDerivePlotVar(d.name()))
    {
      derive_names.push_back(d.name());
      num_derive += d.numDerive();
    }
  }

  int n_data_items = plot_var_map.size() + num_derive;

//----------------------------------------------------------------------modified
#ifdef AMREX_USE_GPIBM
  n_data_items += 2;
#endif
  //------------------------------------------------------------------------------

  // get the time from the first State_Type
  // if the State_Type is ::Interval, this will get t^{n+1/2} instead of t^n
  Real cur_time = state[0].curTime();

  if (level == 0 && ParallelDescriptor::IOProcessor())
  {
    //
    // The first thing we write out is the plotfile type.
    //
    os << thePlotFileType() << '\n';

    if (n_data_items == 0)
      amrex::Error("Must specify at least one valid data item to plot");

    os << n_data_items << '\n';

    //
    // Names of variables
    //
    for (i = 0; i < static_cast<int>(plot_var_map.size()); i++)
    {
      int typ = plot_var_map[i].first;
      int comp = plot_var_map[i].second;
      os << desc_lst[typ].name(comp) << '\n';
    }

    // derived
    for (auto const &dname : derive_names)
    {
      const DeriveRec *rec = derive_lst.get(dname);
      for (i = 0; i < rec->numDerive(); ++i)
      {
        os << rec->variableName(i) << '\n';
      }
    }

//----------------------------------------------------------------------modified
#ifdef AMREX_USE_GPIBM
    os << "sld\n";
    os << "ghs\n";
#endif
    //------------------------------------------------------------------------------

    os << AMREX_SPACEDIM << '\n';
    os << parent->cumTime() << '\n';
    int f_lev = parent->finestLevel();
    os << f_lev << '\n';
    for (i = 0; i < AMREX_SPACEDIM; i++)
      os << Geom().ProbLo(i) << ' ';
    os << '\n';
    for (i = 0; i < AMREX_SPACEDIM; i++)
      os << Geom().ProbHi(i) << ' ';
    os << '\n';
    for (i = 0; i < f_lev; i++)
      os << parent->refRatio(i)[0] << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
      os << parent->Geom(i).Domain() << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
      os << parent->levelSteps(i) << ' ';
    os << '\n';
    for (i = 0; i <= f_lev; i++)
    {
      for (int k = 0; k < AMREX_SPACEDIM; k++)
        os << parent->Geom(i).CellSize()[k] << ' ';
      os << '\n';
    }
    os << (int)Geom().Coord() << '\n';
    os << "0\n"; // Write bndry data.
  }
  // Build the directory to hold the MultiFab at this level.
  // The name is relative to the directory containing the Header file.
  //
  static const std::string BaseName = "/Cell";
  char buf[64];
  snprintf(buf, sizeof buf, "Level_%d", level);
  std::string sLevel = buf;
  //
  // Now for the full pathname of that directory.
  //
  std::string FullPath = dir;
  if (!FullPath.empty() && FullPath[FullPath.size() - 1] != '/')
  {
    FullPath += '/';
  }
  FullPath += sLevel;
  //
  // Only the I/O processor makes the directory if it doesn't already exist.
  //
  if (!levelDirectoryCreated)
  {
    if (ParallelDescriptor::IOProcessor())
    {
      if (!amrex::UtilCreateDirectory(FullPath, 0755))
      {
        amrex::CreateDirectoryFailed(FullPath);
      }
    }
    // Force other processors to wait until directory is built.
    ParallelDescriptor::Barrier();
  }

  if (ParallelDescriptor::IOProcessor())
  {
    os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
    os << parent->levelSteps(level) << '\n';

    for (i = 0; i < grids.size(); ++i)
    {
      RealBox gridloc = RealBox(grids[i], geom.CellSize(), geom.ProbLo());
      for (n = 0; n < AMREX_SPACEDIM; n++)
        os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
    }
    //
    // The full relative pathname of the MultiFabs at this level.
    // The name is relative to the Header file containing this name.
    // It's the name that gets written into the Header.
    //
    if (n_data_items > 0)
    {
      std::string PathNameInHeader = sLevel;
      PathNameInHeader += BaseName;
      os << PathNameInHeader << '\n';
    }

    //----------------------------------------------------------------------modified
    // #ifdef AMREX_USE_EB
    // if (EB2::TopIndexSpaceIfPresent()) {
    //     volfrac threshold for amrvis
    //     if (level == parent->finestLevel()) {
    //         for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
    //             os << "1.0e-6\n";
    //         }
    //     }
    // }
    // #endif
    //------------------------------------------------------------------------------
  }
  //
  // We combine all of the multifabs -- state, derived, etc -- into one
  // multifab -- plotMF.
  int cnt = 0;
  const int nGrow = 0;
  MultiFab plotMF(grids, dmap, n_data_items, nGrow, MFInfo(), Factory());
  MultiFab *this_dat = 0;
  //
  // Cull data from state variables -- use no ghost cells.
  //
  for (i = 0; i < static_cast<int>(plot_var_map.size()); i++)
  {
    int typ = plot_var_map[i].first;
    int comp = plot_var_map[i].second;
    this_dat = &state[typ].newData();
    MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
    cnt++;
  }

  // derived
  if (derive_names.size() > 0)
  {
    for (auto const &dname : derive_names)
    {
      derive(dname, cur_time, plotMF, cnt);
      cnt += derive_lst.get(dname)->numDerive();
    }
  }

//----------------------------------------------------------------------modified
#ifdef AMREX_USE_GPIBM
  plotMF.setVal(0.0, cnt, 2, nGrow);
  IBM::ib.mfa.at(level)->copytoRealMF(plotMF, 0, cnt);
#endif
  //------------------------------------------------------------------------------

  //
  // Use the Full pathname when naming the MultiFab.
  //
  std::string TheFullPath = FullPath;
  TheFullPath += BaseName;
  if (AsyncOut::UseAsyncOut())
  {
    VisMF::AsyncWrite(plotMF, TheFullPath);
  }
  else
  {
    VisMF::Write(plotMF, TheFullPath, how, true);
  }

  levelDirectoryCreated = false; // ---- now that the plotfile is finished
}
