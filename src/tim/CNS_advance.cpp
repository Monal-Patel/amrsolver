#include <CNS.H>
#include <AMReX_FluxRegister.H>
#include <CNS_hydro_K.H>
#include <cns_prob.H>
#ifdef AMREX_USE_GPIBM
#include <IBM.H>
#endif
using namespace amrex;


Real
CNS::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("CNS::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S1 = get_old_data(State_Type);
    MultiFab& S2 = get_new_data(State_Type);

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

  if (order_rk==2) {
  // Low storage SSPRKm2 with m stages (C = m-1, Ceff=1-1/m). Where C is the SSPRK coefficient, it also represents the max CFL over the whole integration step (including m stages). From pg 84 Strong Stability Preserving Runge–kutta And Multistep Time Discretizations
  int m = stages_rk;
  // Copy S2 from S1
  MultiFab::Copy(S2,S1,0,0,NCONS,0);
  // first to m-1 stages
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
  }

  else if (order_rk==3) {
  // Low storage SSPRKm3 with m=n^2, n>=2 stages (C=2, Ceff=0.5). From pg 85 Strong Stability Preserving Runge–kutta And Multistep Time Discretizations
  // FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS);
  // compute_rhs(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
  // MultiFab::LinComb(S2, Real(1.0), S1, 0, dt/2, dSdt, 0, 0, NCONS, 0);

  // FillPatch(*this, Sborder, NGHOST, time + dt/2, State_Type, 0, NCONS);
  // compute_rhs(Sborder, dSdt, dt/2, fr_as_crse, fr_as_fine);
  // MultiFab::Saxpy(S2, dt/2, dSdt, 0, 0, NCONS, 0);

  // FillPatch(*this, Sborder, NGHOST, time + dt, State_Type, 0, NCONS);
  // compute_rhs(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
  // MultiFab::LinComb(S2, Real(2.0)/3, S1, 0, Real(1.0)/3, S2, 0, 0, NCONS, 0);
  // MultiFab::Saxpy(S2, dt/6, dSdt, 0, 0, NCONS, 0);

  // FillPatch(*this, Sborder, NGHOST, time + dt/2, State_Type, 0, NCONS);
  // compute_rhs(Sborder, dSdt, dt/2, fr_as_crse, fr_as_fine);
  // MultiFab::Saxpy(S2, dt/2, dSdt, 0, 0, NCONS, 0);

  // TODO Generally SSPRK(n^2,3) where n>2 - Ceff=1-1/n
  Print() << "SSPRK3 not implemented yet" << std::endl;
  exit(0);

  }

  else if (order_rk==4) {
  Print() << "SSPRK4 not implemented yet" << std::endl;
  exit(0);
  // TODO: SSPRK(10,4) C=6, Ceff=0.6
  }
  return dt;
}


void CNS::compute_rhs (const MultiFab& statemf, MultiFab& dSdt, Real dt,
                   FluxRegister* fr_as_crse, FluxRegister* fr_as_fine)
{
  BL_PROFILE("CNS::compute_rhs()");
    
  //Aux Variables //////////////////////////////////////////////////////////////
  const auto dx     = geom.CellSizeArray();
  const auto dxinv  = geom.InvCellSizeArray();
  Parm const& lparm = *d_parm; // parameters (thermodynamic)
  ProbParm const& lprobparm = *d_prob_parm;

  for (MFIter mfi(statemf, false); mfi.isValid(); ++mfi) {
      auto const& statefab = statemf.array(mfi);
      auto const& prims    = primsmf.array(mfi);
      const Box& bx        = mfi.tilebox();
      const Box& bxg     = mfi.growntilebox(NGHOST);
      amrex::ParallelFor(bxg,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      { cons2prim(i, j, k, statefab, prims, lparm);});
  }
  //////////////////////////////////////////////////////////////////////////////
  Gpu::streamSynchronize(); // ensure primitive variables mf computed before starting mfiter

  //Euler Fluxes ///////////////////////////////////////////////////////////////
  // IMPROVEMENT: Can have pointer function (dynamic casting?) (main euler_flux function) which can be pointed to different flux schemes in the initialisation. The function can pass a parameter struct to include any scheme specfic parameters.
  if(rhs_euler) {
  // weno5js fvs
  if (flux_euler==2) {
    // make multifab for variables
    Array<MultiFab,AMREX_SPACEDIM> pntflxmf;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      pntflxmf[idim].define( statemf.boxArray(), statemf.DistributionMap(), NCONS, NGHOST); pntflxmf[idim] = 0.0_rt; }

    FArrayBox lambdafab; //store eigenvalues max(u+c,u,u-c) in all directions
    // loop over all fabs
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // const Box& bx      = mfi.tilebox();
        const Box& bxg     = mfi.growntilebox(NGHOST);
        const Box& bxnodal = mfi.grownnodaltilebox(-1,0); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

        auto const& statefab = statemf.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                     auto const& nfabfy = numflxmf[1].array(mfi);,
                     auto const& nfabfz = numflxmf[2].array(mfi););

        AMREX_D_TERM(auto const& pfabfx = pntflxmf[0].array(mfi);,
                     auto const& pfabfy = pntflxmf[1].array(mfi);,
                     auto const& pfabfz = pntflxmf[2].array(mfi););

        lambdafab.resize(bxg, 3);
        Elixir lambdagpu   = lambdafab.elixir();
        auto const& lambda = lambdafab.array();

        ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            cons2eulerflux_lambda(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lambda ,lparm);
        });

        // ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        //     cons2char(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lparm);
        // });

        ParallelFor(bxnodal, int(NCONS) , [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
              numericalflux_globallaxsplit(i, j, k, n, statefab ,pfabfx, pfabfy, pfabfz, lambda ,nfabfx, nfabfy, nfabfz); 
        });
    }
  }
  
  // central-split KEEP + JST artificial dissipation
  else if (flux_euler==1) {
  // make multifab for spectral radius and sensor for artificial dissipation
    MultiFab lambdamf; lambdamf.define(statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);
    MultiFab sensormf; sensormf.define(statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);
    lambdamf = 0.0_rt;
    sensormf = 0.0_rt;
    int halfsten = order_keep/2;

    //2 * standard finite difference coefficients
    GpuArray<Real,3> coeffs; coeffs[0]=0.0_rt;coeffs[1]=0.0_rt;coeffs[2]=0.0_rt;
    if (order_keep==4) {
      coeffs[0]=Real(4.0)/3 ,coeffs[1]=Real(-2.0)/12;
    }
    else if (order_keep==6) {
      coeffs[0]=Real(6.0)/4; coeffs[1]=Real(-6.0)/20;coeffs[2]=Real(2.0)/60; 
    }
    else {
      coeffs[0]=Real(1.0);
    }

    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bx      = mfi.tilebox();
      const Box& bxnodal = mfi.grownnodaltilebox(-1,0);
      const Box& bxg     = mfi.growntilebox(NGHOST-1);

      auto const& statefab = statemf.array(mfi);
      auto const& sensor   = sensormf.array(mfi);
      auto const& lambda   = lambdamf.array(mfi);
      auto const& prims    = primsmf.array(mfi);
      AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                    auto const& nfabfy = numflxmf[1].array(mfi);,
                    auto const& nfabfz = numflxmf[2].array(mfi););

      amrex::ParallelFor(bxnodal,  
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        KEEP(i,j,k,halfsten,coeffs,prims,nfabfx,nfabfy,nfabfz,lparm);
      });

      // JST artificial dissipation shock capturing
     amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        ComputeSensorLambda(i,j,k,prims,lambda,sensor,lparm);
      });

      amrex::ParallelFor(bxnodal, NCONS,
      [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
      {
        JSTflux(i,j,k,n,lambda,sensor,statefab,nfabfx,nfabfy,nfabfz);
      });


      }
    }

  // Riemann solver
  else {
    FArrayBox qtmp, slopetmp;
    for (MFIter mfi(statemf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        auto const& sfab = statemf.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                     auto const& nfabfy = numflxmf[1].array(mfi);,
                     auto const& nfabfz = numflxmf[2].array(mfi););

        auto const& q = primsmf.array(mfi);

        const Box& bxg1 = amrex::grow(bx,1);
        slopetmp.resize(bxg1,NCONS);
        Elixir slopeeli = slopetmp.elixir();
        auto const& slope = slopetmp.array();

        // x-direction
        int cdir = 0;
        const Box& xslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(xslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_x(i, j, k, slope, q,lparm);
        });
        const Box& xflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_x(i, j, k, nfabfx, slope, q, lparm);
        });

        // y-direction
        cdir = 1;
        const Box& yslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(yslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_y(i, j, k, slope, q, lparm);
        });
        const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_y(i, j, k, nfabfy, slope, q, lparm);
        });

        // z-direction
        cdir = 2;
        const Box& zslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(zslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_z(i, j, k, slope, q, lparm);
        });
        const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_z(i, j, k, nfabfz, slope, q, lparm);
        });

        // don't have to do this, but we could
        // qeli.clear(); // don't need them anymore
        slopeeli.clear();
    }
  }
  } 
  //////////////////////////////////////////////////////////////////////////////


  // Viscous Fluxes ////////////////////////////////////////////////////////////
  // Gpu::streamSynchronize(); // ensure all rhs terms computed before assembly
  // We have a separate MFIter loop here than the Euler fluxes and the source terms, so the work can be further parallised. As different MFIter loops can be in different GPU streams. 

  // Although conservative FD (finite difference) derivatives of viscous fluxes are not requried in the boundary layer, standard FD are likely sufficient. However, considering grid and flow discontinuities (coarse-interface flux-refluxing and viscous derivatives near shocks), conservative FD derivatives are preferred.
  if (rhs_visc) {
    // Make multifab for derivatives of primitive variables and viscous fluxes
    // Array<MultiFab,AMREX_SPACEDIM> pntvflxmf;
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //   pntvflxmf[idim].define(statemf.boxArray(), statemf.DistributionMap(), NCONS, NGHOST); pntvflxmf[idim] = 0.0_rt;}

    // loop over all fabs
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bxpflx  = mfi.growntilebox(1);
        const Box& bxnodal  = mfi.grownnodaltilebox(-1,0); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

        auto const& prims    = primsmf.array(mfi);

        AMREX_D_TERM(auto const& pfabfx = pntvflxmf[0].array(mfi);,
                    auto const& pfabfy = pntvflxmf[1].array(mfi);,
                    auto const& pfabfz = pntvflxmf[2].array(mfi););

        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                    auto const& nfabfy = numflxmf[1].array(mfi);,
                    auto const& nfabfz = numflxmf[2].array(mfi););

        // Compute transport coefficients
        // amrex::ParallelFor(bxg,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     cons2prim(i, j, k, statefab, prims, *lparm);
        //     // prim2tran(i, j, k, prims, trans, *lparm);
        // });

        // compute u,v,w,T derivatives and compute physical viscous fluxes
        amrex::ParallelFor(bxpflx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            viscfluxes(i, j, k, prims, pfabfx, pfabfy, pfabfz, dxinv, lparm);
        });

        // compute numerical viscous fluxes
        amrex::ParallelFor(bxnodal, NCONS,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
            visc_numericalfluxes(i, j, k, n, pfabfx, pfabfy, pfabfz, nfabfx, nfabfy, nfabfz);
        });
      }
  }
  //////////////////////////////////////////////////////////////////////////////

  // Re-fluxing ////////////////////////////////////////////////////////////////
  // Must be done before adding source terms.
  if (fr_as_crse) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
          const Real scale = -dt*dA;
          fr_as_crse->CrseInit(numflxmf[idim], idim, 0, 0, NCONS, scale, FluxRegister::ADD);
      }
  }
  if (fr_as_fine) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
          const Real scale = dt*dA;
          fr_as_fine->FineAdd(numflxmf[idim], idim, 0, 0, NCONS, scale);
      }
  }
  //////////////////////////////////////////////////////////////////////////////

  // Assemble RHS //////////////////////////////////////////////////////////////
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

  // Add source term ///////////////////////////////////////////////////////////
  if (rhs_source) {
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const Box& bx   = mfi.tilebox();
        auto const& dsdtfab = dSdt.array(mfi);
        auto const& statefab = statemf.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        { user_source(i,j,k,statefab,dsdtfab,lprobparm); });
    }
  }
  //////////////////////////////////////////////////////////////////////////////

}