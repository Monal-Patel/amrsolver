#include <CNS.h>
#include <CNS_K.h>
#include <prob.h>

using namespace amrex;

int CNS::num_state_data_types = 0;

static Box the_same_box(const Box& b) { return b; }
// static Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

using BndryFunc = StateDescriptor::BndryFunc;

//
// Components are:
//  Interior, Inflow, Outflow, Symmetry, SlipWall, NoSlipWall, User defined
//  (0)         (1)     (2)       (3)      (4)        (5)         (6)
static int scalar_bc[] = {BCType::int_dir,      BCType::ext_dir,
                          BCType::foextrap,     BCType::reflect_even,
                          BCType::reflect_even, BCType::reflect_even,
                          BCType::ext_dir};

static int norm_vel_bc[] = {BCType::int_dir,     BCType::ext_dir,
                            BCType::foextrap,    BCType::reflect_odd,
                            BCType::reflect_odd, BCType::reflect_odd,
                            BCType::ext_dir};

static int tang_vel_bc[] = {BCType::int_dir,      BCType::ext_dir,
                            BCType::foextrap,     BCType::reflect_even,
                            BCType::reflect_even, BCType::reflect_odd,
                            BCType::ext_dir};

static void set_scalar_bc(BCRec& bc, const BCRec* phys_bc) {
  const int* lo_bc = phys_bc->lo();
  const int* hi_bc = phys_bc->hi();
  for (int i = 0; i < AMREX_SPACEDIM; i++) {
    bc.setLo(i, scalar_bc[lo_bc[i]]);
    bc.setHi(i, scalar_bc[hi_bc[i]]);
  }
}

static void set_x_vel_bc(BCRec& bc, const BCRec* phys_bc) {
  const int* lo_bc = phys_bc->lo();
  const int* hi_bc = phys_bc->hi();
  bc.setLo(0, norm_vel_bc[lo_bc[0]]);
  bc.setHi(0, norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
  bc.setLo(1, tang_vel_bc[lo_bc[1]]);
  bc.setHi(1, tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
  bc.setLo(2, tang_vel_bc[lo_bc[2]]);
  bc.setHi(2, tang_vel_bc[hi_bc[2]]);
#endif
}

static void set_y_vel_bc(BCRec& bc, const BCRec* phys_bc) {
  const int* lo_bc = phys_bc->lo();
  const int* hi_bc = phys_bc->hi();
  bc.setLo(0, tang_vel_bc[lo_bc[0]]);
  bc.setHi(0, tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
  bc.setLo(1, norm_vel_bc[lo_bc[1]]);
  bc.setHi(1, norm_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
  bc.setLo(2, tang_vel_bc[lo_bc[2]]);
  bc.setHi(2, tang_vel_bc[hi_bc[2]]);
#endif
}

static void set_z_vel_bc(BCRec& bc, const BCRec* phys_bc) {
  const int* lo_bc = phys_bc->lo();
  const int* hi_bc = phys_bc->hi();
  bc.setLo(0, tang_vel_bc[lo_bc[0]]);
  bc.setHi(0, tang_vel_bc[hi_bc[0]]);
  bc.setLo(1, tang_vel_bc[lo_bc[1]]);
  bc.setHi(1, tang_vel_bc[hi_bc[1]]);
  bc.setLo(2, norm_vel_bc[lo_bc[2]]);
  bc.setHi(2, norm_vel_bc[hi_bc[2]]);
}

void CNS::variableSetUp() {
  // Closures and Problem structures (available on both CPU and GPU)
  CNS::h_prob_closures = new PROB::ProbClosures{};
  CNS::h_prob_parm = new PROB::ProbParm{};
  CNS::h_phys_bc = new BCRec{};
#ifdef AMREX_USE_GPU
  CNS::d_prob_closures =
      (PROB::ProbClosures*)The_Arena()->alloc(sizeof(PROB::ProbClosures));
  CNS::d_prob_parm =
      (PROB::ProbParm*)The_Arena()->alloc(sizeof(PROB::ProbParm));
  CNS::d_phys_bc = (BCRec*)The_Arena()->alloc(sizeof(BCRec));
#else
  CNS::d_prob_closures = h_prob_closures;
  CNS::d_prob_parm = h_prob_parm;
  CNS::d_phys_bc = h_phys_bc;
#endif

  // Read input parameters
  read_params();

  // Independent (solved) variables and their boundary condition types
  bool state_data_extrap = false;
  bool store_in_checkpoint = true;
  desc_lst.addDescriptor(State_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, h_prob_closures->NGHOST, h_prob_closures->NCONS, &lincc_interp,
                         state_data_extrap, store_in_checkpoint);
  // https://github.com/AMReX-Codes/amrex/issues/396

  Vector<BCRec> bcs(PROB::ProbClosures::NCONS);

  // Physical boundary conditions ////////////////////////////////////////////

  // TODO assert Ncons = cons_vars_type.length and cons_vars_names_length 
  for (int cnt=0;cnt<h_prob_closures->NCONS;cnt++) {
    if (PROB::cons_vars_type[cnt]==0) {
      // Print() << cnt << " " << PROB::cons_vars_names[cnt] << std::endl;
      set_scalar_bc(bcs[cnt], h_phys_bc);
    }
    else if (PROB::cons_vars_type[cnt]==1) {
      // Print() << cnt << " " << PROB::cons_vars_names[cnt] << std::endl;
      set_x_vel_bc(bcs[cnt], h_phys_bc);
    }
    else if (PROB::cons_vars_type[cnt]==2) {
      // Print() << cnt << " " << PROB::cons_vars_names[cnt] << std::endl;
      set_y_vel_bc(bcs[cnt], h_phys_bc);
    }
    else if (PROB::cons_vars_type[cnt]==3) {
      // Print() << cnt << " " << PROB::cons_vars_names[cnt] << std::endl;
      set_z_vel_bc(bcs[cnt], h_phys_bc);
    }
  }
  // Boundary conditions
  StateDescriptor::BndryFunc bndryfunc(cns_bcfill);
  StateDescriptor::setBndryFuncThreadSafety(true);
  bndryfunc.setRunOnGPU(true);
  // applies bndry func to all variables in desc_lst starting from from 0.
  desc_lst.setComponent(State_Type, 0, PROB::cons_vars_names, bcs, bndryfunc);

  num_state_data_types = desc_lst.size();
  ////////////////////////////////////////////////////////////////////////////

  // Define derived quantities ///////////////////////////////////////////////

  // Can have derpres, dertemp, derprimvar defined in prob.h
  // Can we avoid this by modifying write plot file routine? --> AMReX not designed for this, not trivial.
  // 

  // PROB::
  // Pressure
  // derive_lst.add("Pressure", IndexType::TheCellType(), 1,derpres,
                //  the_same_box);
  // printf("address:%p \n",&PROB::ProbClosures::derpres);
  // std::cout << typeid(derpres).name() << std::endl;
  // std::cout << typeid(&PROB::ProbClosures::derpres).name() << std::endl;

  // Vector<std::string> derived_vars_names={"Pressure","Temperature"};

  // derive_lst.add("Pressure", IndexType::TheCellType(), 1, derpres, the_same_box);

  // derive_lst.addComponent("Pressure", desc_lst, State_Type, h_prob_closures->URHO, h_prob_closures->NCONS);

  // // Temperature
  // derive_lst.add("Temperature", IndexType::TheCellType(), 1, dertemp,
  //                the_same_box);
  // derive_lst.addComponent("Temperature", desc_lst, State_Type, h_prob_closures->URHO, h_prob_closures->NCONS);

}