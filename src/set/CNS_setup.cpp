#include <CNS.H>
#include <CNS_index_macros.H>
#include "CNS_derive.H"

using namespace amrex;

int CNS::num_state_data_types = 0;
Parm* CNS::h_parm = nullptr;
Parm* CNS::d_parm = nullptr;
ProbParm* CNS::h_prob_parm = nullptr;
ProbParm* CNS::d_prob_parm = nullptr;

static Box the_same_box (const Box& b) { return b; }
//static Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

using BndryFunc = StateDescriptor::BndryFunc;

//
// Components are:
//  Interior, Inflow, Outflow, Symmetry, SlipWall, NoSlipWall, User defined 
//
static int scalar_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_even, BCType::ext_dir
};

static int norm_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_odd,  BCType::reflect_odd,  BCType::reflect_odd, BCType::ext_dir
};

static int tang_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_odd, BCType::ext_dir
};

static void set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        bc.setLo(i,scalar_bc[lo_bc[i]]);
        bc.setHi(i,scalar_bc[hi_bc[i]]);
    }
}

static
void
set_x_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,norm_vel_bc[lo_bc[0]]);
    bc.setHi(0,norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_y_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,norm_vel_bc[lo_bc[1]]);
    bc.setHi(1,norm_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

#if (AMREX_SPACEDIM == 3)
static
void
set_z_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
    bc.setLo(2,norm_vel_bc[lo_bc[2]]);
    bc.setHi(2,norm_vel_bc[hi_bc[2]]);
}
#endif

void
CNS::variableSetUp ()
{
  // Since this is a GPU/CPU code - it is useful to have some basic parameters always available on both CPU and GPU always.
  // These host and device variables are deleted in CNS::variableCleanUp().
    CNS::h_parm = new Parm{};
    CNS::h_prob_parm = new ProbParm{};
#ifdef AMREX_USE_GPU
    CNS::d_parm = (Parm*)The_Arena()->alloc(sizeof(Parm));
    CNS::d_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));
#else
    CNS::d_parm      = h_parm;
    CNS::d_prob_parm = h_prob_parm;
#endif

    read_params();

    bool state_data_extrap = false;
    bool store_in_checkpoint = true;
    desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
                           StateDescriptor::Point,NGHOST,NCONS,
                           &cell_cons_interp,state_data_extrap,store_in_checkpoint);

    Vector<BCRec>       bcs(NCONS);
    Vector<std::string> name(NCONS);

    // Physical boundary conditions ////////////////////////////////////////////
    int cnt = 0;
    set_scalar_bc(bcs[cnt],phys_bc); 
    name[cnt] = "density";

    cnt++; 
    set_x_vel_bc(bcs[cnt],phys_bc);  
    name[cnt] = "xmom";

    cnt++; 
    set_y_vel_bc(bcs[cnt],phys_bc);
    name[cnt] = "ymom";

#if (AMREX_SPACEDIM == 3)
    cnt++;
    set_z_vel_bc(bcs[cnt],phys_bc);
    name[cnt] = "zmom";
#endif
    cnt++; 
    set_scalar_bc(bcs[cnt],phys_bc);
    name[cnt] = "rho_et";

    StateDescriptor::BndryFunc bndryfunc(cns_bcfill);
    bndryfunc.setRunOnGPU(true);


    // applies bndry func to all variables in desc_lst starting from from URHO (0).
    desc_lst.setComponent(State_Type, URHO,name, bcs, bndryfunc);


    num_state_data_types = desc_lst.size();

    StateDescriptor::setBndryFuncThreadSafety(true);
    ////////////////////////////////////////////////////////////////////////////

    // Define derived quantities ///////////////////////////////////////////////
    // Pressure
    derive_lst.add("pressure",IndexType::TheCellType(),1,
                   derpres,the_same_box);
    derive_lst.addComponent("pressure",desc_lst,State_Type,URHO,NCONS);

    // Temperature
    derive_lst.add("temperature", IndexType::TheCellType(), 1,
                 dertemp,the_same_box);
    derive_lst.addComponent("temperature", desc_lst, State_Type, URHO,NCONS);

    // Velocities
    derive_lst.add("x_velocity", amrex::IndexType::TheCellType(), 1, dervel, the_same_box);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,Density,1);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,Xmom,1);

    derive_lst.add("y_velocity", amrex::IndexType::TheCellType(), 1, dervel, the_same_box);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,Density,1);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,Ymom,1);

#if (AMREX_SPACEDIM == 3)
    derive_lst.add("z_velocity", amrex::IndexType::TheCellType(), 1, dervel, the_same_box);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,Density,1);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,Zmom,1);
#endif
    // desc_lst.addDescriptor(Cost_Type, IndexType::TheCellType(), StateDescriptor::Point,
    //                        0,1, &pc_interp);
    // desc_lst.setComponent(Cost_Type, 0, "Cost", bc, bndryfunc);

    ////////////////////////////////////////////////////////////////////////////
}

void
CNS::variableCleanUp ()
{
    delete h_parm;
    delete h_prob_parm;

#ifdef AMREX_USE_GPU
    The_Arena()->free(d_parm);
    The_Arena()->free(d_prob_parm);
// #else
//     delete d_parm;
//     delete d_prob_parm;
#endif
    desc_lst.clear();
    derive_lst.clear();

// #ifdef AMREX_USE_GPU
//     The_Arena()->free(dp_refine_boxes);
// #endif
}
