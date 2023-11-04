#include <CNS.h>
#include <AMReX_FluxRegister.H>
#include <CNS_hydro_K.h>
#include <prob.h>
#include <CentralKEEP.h>
#include <Riemann.h>
#include <High_resolution.h>
#include <NLDE.h>
#ifdef AMREX_USE_GPIBM
#include <IBM.h>
#endif
using namespace amrex;


Real CNS::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("CNS::advance()");

    // Print() << "-------- advance start -------" << std::endl;
    // Print() << "time = " << time << std::endl;
    // Print() << "dt = " << dt << std::endl;
    // state[0].printTimeInterval(std::cout);
    // Print() << "---------------------------------" << std::endl;

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S1 = get_old_data(State_Type);
    MultiFab& S2 = get_new_data(State_Type);

    MultiFab& dSdt = VdSdt[level];
    MultiFab& Sborder = VSborder[level];

    FluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        CNS& fine_level = getLevel(level+1);
        fr_as_crse = fine_level.flux_reg.get();
    }

    FluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = flux_reg.get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(Real(0.0));
    }


  if (order_rk==-2) {
    // Original time integration ///////////////////////////////////////////////
    // RK2 stage 1
    FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, Real(0.5)*dt, fr_as_crse, fr_as_fine);
    // U^* = U^n + dt*dUdt^n
    MultiFab::LinComb(S2, Real(1.0), Sborder, 0, dt, dSdt, 0, 0, NCONS, 0);

    // RK2 stage 2
    // After fillpatch Sborder = U^n+dt*dUdt^n
    state[0].setNewTimeLevel (time+dt);
    FillPatch(*this, Sborder, NGHOST, time+dt, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, Real(0.5)*dt, fr_as_crse, fr_as_fine);
    // S_new = 0.5*(Sborder+S_old) = U^n + 0.5*dt*dUdt^n
    MultiFab::LinComb(S2, Real(0.5), Sborder, 0, Real(0.5), S1, 0, 0, NCONS, 0);
    // S_new += 0.5*dt*dSdt
    MultiFab::Saxpy(S2, Real(0.5)*dt, dSdt, 0, 0, NCONS, 0);
    // We now have S_new = U^{n+1} = (U^n+0.5*dt*dUdt^n) + 0.5*dt*dUdt^*
    ////////////////////////////////////////////////////////////////////////////
  }
  else if (order_rk==0) { // returns rhs
    FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
    MultiFab::Copy(S2,dSdt,0,0,NCONS,0);
  }
  else if (order_rk==1) {
    FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS); // filled at t_n to evalulate f(t_n,y_n).
    compute_rhs(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
    MultiFab::LinComb(S2, Real(1.0), S1, 0, dt, dSdt, 0, 0, NCONS, 0);
  }
  else if (order_rk==2) {
  // Low storage SSPRKm2 with m stages (C = m-1, Ceff=1-1/m). Where C is the SSPRK coefficient, it also represents the max CFL over the whole integration step (including m stages). From pg 84 Strong Stability Preserving Runge–kutta And Multistep Time Discretizations
  int m = stages_rk;
  // Copy S2 from S1
  MultiFab::Copy(S2,S1,0,0,NCONS,0);
  state[0].setOldTimeLevel (time);
  state[0].setNewTimeLevel (time);
  // first to m-1 stages
  // Print() << "-------- before RK stages -------" << std::endl;
  // Print() << "time = " << time << std::endl;
  // Print() << "dt = " << dt << std::endl;
  // state[0].printTimeInterval(std::cout);
  // Print() << "---------------------------------" << std::endl;
  for (int i=1; i<=m-1; i++) {
    FillPatch(*this, Sborder, NGHOST, time + dt*Real(i-1)/(m-1) , State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt/Real(m-1), fr_as_crse, fr_as_fine);
    MultiFab::Saxpy(S2, dt/Real(m-1), dSdt, 0, 0, NCONS, 0);
    state[State_Type].setNewTimeLevel(time + dt*Real(i)/(m-1)); // important to do this for correct fillpatch interpolations for the proceeding stages
  }
  // final stage
  FillPatch(*this, Sborder, NGHOST, time + dt, State_Type, 0, NCONS);
  compute_rhs(Sborder, dSdt, dt/Real(m-1), fr_as_crse, fr_as_fine);
  MultiFab::LinComb(S2, Real(m-1), S2, 0, dt, dSdt, 0, 0, NCONS, 0);
  MultiFab::LinComb(S2, Real(1.0)/m, S1, 0, Real(1.0)/m, S2, 0, 0, NCONS, 0);

  state[State_Type].setNewTimeLevel(time + dt); // important to do this for correct fillpatch interpolations for the proceeding stages


  // Print() << "--------- after RK stages --------" << std::endl;
  // Print() << "time = " << time << std::endl;
  // Print() << "dt = " << dt << std::endl;
  // state[0].printTimeInterval(std::cout);
  // Print() << "----------------------------------" << std::endl;
  }

  else if (order_rk==3) {

  if (stages_rk==3) {
    state[0].setOldTimeLevel (time);
    // http://ketch.github.io/numipedia/methods/SSPRK33.html
    // state[0].setOldTimeLevel (time);
    FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS); // filled at t_n to evalulate f(t_n,y_n).
    compute_rhs(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
    MultiFab::LinComb(S2, Real(1.0), S1, 0, dt, dSdt, 0, 0, NCONS, 0);

    state[0].setNewTimeLevel (time+dt); // same time as upcoming FillPatch ensures we copy S2 to Sborder, without time interpolation
    FillPatch(*this, Sborder, NGHOST, time+dt, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt/4, fr_as_crse, fr_as_fine);
    MultiFab::Xpay(dSdt, dt, S2, 0, 0, NCONS, 0);
    MultiFab::LinComb(S2, Real(3.0)/4, S1, 0, Real(1.0)/4, dSdt, 0, 0, NCONS, 0);

    state[0].setNewTimeLevel (time+dt/2);// same time as upcoming FillPatch ensures we copy S2 to Sborder, without time interpolation
    FillPatch(*this, Sborder, NGHOST, time + dt/2, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt*Real(2.0)/3, fr_as_crse, fr_as_fine);
    MultiFab::Xpay(dSdt, dt, S2, 0, 0, NCONS, 0);
    MultiFab::LinComb(S2, Real(1.0)/3, S1, 0, Real(2.0)/3, dSdt, 0, 0, NCONS, 0);

    state[State_Type].setNewTimeLevel(time + dt); // important to do this for correct fillpatch interpolations for the proceeding stages
  }

  else if (stages_rk==4) {
    // http://ketch.github.io/numipedia/methods/SSPRK43.html and From pg 85 Strong Stability Preserving Runge–kutta And Multistep Time Discretizations

    state[0].setOldTimeLevel (time);
    FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt/2, fr_as_crse, fr_as_fine);
    MultiFab::LinComb(S2, Real(1.0), S1, 0, dt/2, dSdt, 0, 0, NCONS, 0);

    state[0].setNewTimeLevel (time+dt/2); // same time as upcoming FillPatch ensures we copy S2 to Sborder, without time interpolation
    FillPatch(*this, Sborder, NGHOST, time + dt/2, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt/2, fr_as_crse, fr_as_fine);
    MultiFab::Saxpy(S2, dt/2, dSdt, 0, 0, NCONS, 0);

    state[0].setNewTimeLevel (time+dt); // same time as upcoming FillPatch ensures we copy S2 to Sborder, without time interpolation
    FillPatch(*this, Sborder, NGHOST, time + dt, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt/6, fr_as_crse, fr_as_fine);
    MultiFab::LinComb(S2, Real(2.0)/3, S1, 0, Real(1.0)/3, S2, 0, 0, NCONS, 0);
    MultiFab::Saxpy(S2, dt/6, dSdt, 0, 0, NCONS, 0);

    state[0].setNewTimeLevel (time+dt/2); // same time as upcoming FillPatch ensures we copy S2 to Sborder, without time interpolation
    FillPatch(*this, Sborder, NGHOST, time + dt/2, State_Type, 0, NCONS);
    compute_rhs(Sborder, dSdt, dt/2, fr_as_crse, fr_as_fine);
    MultiFab::Saxpy(S2, dt/2, dSdt, 0, 0, NCONS, 0);

    state[State_Type].setNewTimeLevel(time + dt); // important to do this for correct fillpatch interpolations for the proceeding stages
  }

  else {
    // Low storage SSPRKm3 with m=n^2, n>=3 stages (C=2, Ceff=0.5). From pg 85 Strong Stability Preserving Runge–kutta And Multistep Time Discretizations
    // TODO Generally SSPRK(n^2,3) where n>2 - Ceff=1-1/n
    Print() << "SSPRK(m^2)3 not implemented yet" << std::endl;
    exit(0);
  }

  }

  else if (order_rk==4) {
    Print() << "SSPRK4 not implemented yet" << std::endl;
    exit(0);
    // TODO: SSPRK(10,4) C=6, Ceff=0.6
  }
  return dt;
}


void CNS::compute_rhs (MultiFab& statemf, MultiFab& dSdt, Real dt,
                   FluxRegister* fr_as_crse, FluxRegister* fr_as_fine)
{
  BL_PROFILE("CNS::compute_rhs()");
    
  //Aux Variables //////////////////////////////////////////////////////////////
  const auto dx     = geom.CellSizeArray();
  const auto dxinv  = geom.InvCellSizeArray();
  PROB::ProbClosures& lclosures = *CNS::d_prob_closures;
  PROB::ProbParm const& lprobparm = *d_prob_parm;
  MultiFab& primsmf  = Vprimsmf[level];
  Array<MultiFab,AMREX_SPACEDIM>& numflxmf = Vnumflxmf[level];

  if constexpr (PROB::do_pde==1) {
    NLDE::cons2prim(level, statemf, primsmf, lclosures);
  }
  else{
    for (MFIter mfi(statemf, false); mfi.isValid(); ++mfi) {
      auto const& statefab = statemf.array(mfi);
      auto const& prims    = primsmf.array(mfi);
      const Box& bxg       = mfi.growntilebox(NGHOST);
      amrex::ParallelFor(bxg,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      { cons2prim(i, j, k, statefab, prims, lclosures);});}
  }
  //////////////////////////////////////////////////////////////////////////////
  Gpu::streamSynchronize(); // ensure primitive variables mf computed before starting mfiter
  
  // Immersed boundaries ///////////////////////////////////////////////////////
  // Compute on CPU always
#ifdef AMREX_USE_GPIBM
  IBM::IBMultiFab& ibmf = *IBM::ibm.ibMFa[level];
  IBM::ibm.computeGPs(level, statemf, primsmf, ibmf, lclosures);
#endif
  //////////////////////////////////////////////////////////////////////////////

  //Euler Fluxes ///////////////////////////////////////////////////////////////
  if(rhs_euler) {
    if constexpr (PROB::do_pde==1) {NLDE::eflux(level,statemf,primsmf,numflxmf);}
    else { 
      if (flux_euler==2) {HiRes::FluxWENO(statemf,primsmf,numflxmf);  }
      else if (flux_euler==1) {CentralKEEP::eflux(statemf,primsmf,numflxmf);}
      else {Riemann::Flux(statemf,primsmf,numflxmf);}
    }

    // Euler flux corrections (overwrite numflxmf) //
    // Recompute fluxes on planes adjacent to physical boundaries (Order reduction)
    if (flux_euler==1 && !(CentralKEEP::order_keep==2)) {
      CentralKEEP::Flux_2nd_Order_KEEP(geom,primsmf,numflxmf);
      // Order reduction near IBM
#if AMREX_USE_GPIBM
    IBM::IBMultiFab& ibmf = *IBM::ibm.ibMFa[level];
    CentralKEEP::ibm_flux_correction(primsmf,numflxmf,ibmf);
#endif
    }



    // Artificial dissipation (adding to numflxmf)
    // JST artificial dissipation shock capturing
    if (art_diss==1) {
    // make multifab for spectral radius and sensor for artificial dissipation
      MultiFab lambdamf; lambdamf.define(statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);
      MultiFab sensormf; sensormf.define(statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);
      lambdamf = 0.0_rt;
      sensormf = 0.0_rt;
      for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& bx      = mfi.tilebox();
        const Box& bxnodal = mfi.grownnodaltilebox(-1,0);

        auto const& statefab = statemf.array(mfi);
        auto const& sensor   = sensormf.array(mfi);
        auto const& lambda   = lambdamf.array(mfi);
        auto const& prims    = primsmf.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                      auto const& nfabfy = numflxmf[1].array(mfi);,
                      auto const& nfabfz = numflxmf[2].array(mfi););
        amrex::ParallelFor(bx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            ComputeSensorLambda(i,j,k,prims,lambda,sensor,lclosures);
          });

          amrex::ParallelFor(bxnodal, NCONS,
          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
          {
            JSTflux(i,j,k,n,lambda,sensor,statefab,nfabfx,nfabfy,nfabfz,lclosures);
          }); 
      }
    }
  } 

  //////////////////////////////////////////////////////////////////////////////


  // Viscous Fluxes ////////////////////////////////////////////////////////////
  // Gpu::streamSynchronize(); // ensure all rhs terms computed before assembly
  // We have a separate MFIter loop here than the Euler fluxes and the source terms, so the work can be further parallised. As different MFIter loops can be in different GPU streams. 

  // Although conservative FD (finite difference) derivatives of viscous fluxes are not requried in the boundary layer, standard FD are likely sufficient. However, considering grid and flow discontinuities (coarse-interface flux-refluxing and viscous derivatives near shocks), conservative FD derivatives are preferred.
  if (rhs_visc) {
    Array<MultiFab,AMREX_SPACEDIM>& pntvflxmf = Vpntvflxmf[level];
#if AMREX_USE_GPIBM
    IBM::IBMultiFab& ibMultiFab = *IBM::ibm.ibMFa[level];
#endif
    //for each fab
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bxpflx  = mfi.growntilebox(1);
      auto const& prims  = primsmf.array(mfi);

      AMREX_D_TERM(auto const& pfabfx = pntvflxmf[0].array(mfi);,
                  auto const& pfabfy  = pntvflxmf[1].array(mfi);,
                  auto const& pfabfz  = pntvflxmf[2].array(mfi););

      AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                  auto const& nfabfy  = numflxmf[1].array(mfi);,
                  auto const& nfabfz  = numflxmf[2].array(mfi););

      // compute u,v,w,T derivatives and compute physical viscous fluxes
      amrex::ParallelFor(bxpflx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        ViscousFluxes(i, j, k, prims, pfabfx, pfabfy, pfabfz, dxinv, lclosures);
      });

      // Physical boundary viscous flux corrections 
      // (overwrite pfabfx, pfabfy, pfabfz)
      // TODO:: generalise to wall boundary in x and z directions
      const Box& bx  = mfi.tilebox();
      //xlo
      //xhi

      //ylo
      if(geom.Domain().smallEnd(1)==bx.smallEnd(1)) {
        if ((*h_phys_bc).lo(1)==6) {
          int jj = bx.smallEnd(1)-1;
          IntVect small = {bxpflx.smallEnd(0), jj, bxpflx.smallEnd(2)};
          IntVect big   = {bxpflx.bigEnd(0)  , jj, bxpflx.bigEnd(2)  };
          Box bxboundary(small,big);

          amrex::ParallelFor(bxboundary,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
              ViscousWallFluxes(i, j, k, 0, prims, pfabfx, pfabfy, pfabfz, dxinv, lclosures);
          });
        }
      }
      //yhi
      if(geom.Domain().bigEnd(1)==bx.bigEnd(1)) {
        if ((*h_phys_bc).hi(1)==6) {
          int jj = bx.bigEnd(1) + 1;
          IntVect small = {bxpflx.smallEnd(0), jj, bxpflx.smallEnd(2)};
          IntVect big   = {bxpflx.bigEnd(0)  , jj, bxpflx.bigEnd(2)  };
          Box bxboundary(small,big);

          amrex::ParallelFor(bxboundary,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
              ViscousWallFluxes(i, j, k, 1, prims, pfabfx, pfabfy, pfabfz, dxinv, lclosures);
          });
        }
      }
      //zlo
      //zhi

#if AMREX_USE_GPIBM
      //IBM GP viscous flux correction
      auto const& ibFab = ibMultiFab.get(mfi);
      auto const& markers = ibMultiFab.array(mfi);
      auto const gp_ijk   = ibFab.gpData.gp_ijk.data();
      amrex::ParallelFor(ibFab.gpData.ngps, [=] AMREX_GPU_DEVICE (int ii)
      {
        ViscousFluxGP(gp_ijk[ii](0),gp_ijk[ii](1),gp_ijk[ii](2),markers,prims,pfabfx,pfabfy,pfabfz,dxinv,lclosures);
      });
#endif

      // compute numerical viscous fluxes (add to numflxmf)
      const Box& bxnodal  = mfi.grownnodaltilebox(-1,0); // extent is 0,N+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same
      amrex::ParallelFor(bxnodal, NCONS,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
        ViscousNumericalFluxes(i, j, k, n, pfabfx, pfabfy, pfabfz, nfabfx, nfabfy, nfabfz);
      });
    }
  }

  // Re-fluxing ////////////////////////////////////////////////////////////////
  // if (do_reflux) {
  // if (fr_as_crse) {
  //     for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
  //         const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
  //         const Real scale = -dt*dA;
  //         fr_as_crse->CrseInit(numflxmf[idim], idim, 0, 0, NCONS, scale, FluxRegister::ADD);
  //     }
  // }
  // if (fr_as_fine) {
  //     for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
  //         const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
  //         const Real scale = dt*dA;
  //         fr_as_fine->FineAdd(numflxmf[idim], idim, 0, 0, NCONS, scale);
  //     }
  // }
  // }
  //////////////////////////////////////////////////////////////////////////////

  // Assemble RHS fluxes ///////////////////////////////////////////////////////
  Gpu::streamSynchronize(); // ensure all fluxes computed before assembly
  for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi){
    const Box& bx   = mfi.tilebox();
    auto const& dsdtfab = dSdt.array(mfi);
    AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                  auto const& nfabfy = numflxmf[1].array(mfi);,
                  auto const& nfabfz = numflxmf[2].array(mfi););

    // add euler and viscous derivatives to rhs
    amrex::ParallelFor(bx, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
    dsdtfab(i,j,k,n) = dxinv[0]*(nfabfx(i,j,k,n) - nfabfx(i+1,j,k,n))
    +           dxinv[1] *(nfabfy(i,j,k,n) - nfabfy(i,j+1,k,n))
    +           dxinv[2] *(nfabfz(i,j,k,n) - nfabfz(i,j,k+1,n));
    });
  }


  //////////////////////////////////////////////////////////////////////////////

  // Add source term to RHS ////////////////////////////////////////////////////
  if (rhs_source) {
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi){
      const Box& bx   = mfi.tilebox();
      auto const& dsdtfab = dSdt.array(mfi);
      auto const& statefab = statemf.array(mfi);

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      { user_source(i,j,k,statefab,dsdtfab,lprobparm,lclosures,dx); });
    }
  }
  //////////////////////////////////////////////////////////////////////////////

  // Set solid point RHS to 0 //////////////////////////////////////////////////
#if AMREX_USE_GPIBM
  IBM::IBMultiFab& mfab = *IBM::ibm.ibMFa[level];
  for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi){
    const Box& bx   = mfi.tilebox();
    auto const& dsdtfab = dSdt.array(mfi);
    IBM::IBFab &fab = mfab.get(mfi);
    Array4<bool> ibMarkers = fab.array();
    amrex::ParallelFor(bx, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
    dsdtfab(i,j,k,n) = dsdtfab(i,j,k,n)*(1 - int(ibMarkers(i,j,k,0)));
    });
  }
#endif
  //////////////////////////////////////////////////////////////////////////////

}