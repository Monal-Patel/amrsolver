#include <CNS.H>
#include <AMReX_FluxRegister.H>
#include <CNS_hydro_K.H>
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

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    MultiFab dSdt(grids,dmap,NCONS,0,MFInfo(),Factory());
    MultiFab Sborder(grids,dmap,NCONS,NGHOST,MFInfo(),Factory());

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


    // RK2 stage 1
    // After fillpatch Sborder = U^n
    FillPatch(*this, Sborder, NGHOST, time, State_Type, 0, NCONS);
    // fillpatch copies data from leveldata to sborder

#ifdef AMREX_USE_GPIBM
    IBM::ib.computeGPs(level,Sborder);
    // exit(0);
#endif

    compute_rhs(Sborder, dSdt, Real(0.5)*dt, fr_as_crse, fr_as_fine);
    // U^* = U^n + dt*dUdt^n
    MultiFab::LinComb(S_new, Real(1.0), Sborder, 0, dt, dSdt, 0, 0, NCONS, 0);

    // RK2 stage 2
    // After fillpatch Sborder = U^n+dt*dUdt^n
    FillPatch(*this, Sborder, NGHOST, time+dt, State_Type, 0, NCONS);

#ifdef AMREX_USE_GPIBM
    IBM::ib.computeGPs(level,Sborder);
#endif

    compute_rhs(Sborder, dSdt, Real(0.5)*dt, fr_as_crse, fr_as_fine);
    // S_new = 0.5*(Sborder+S_old) = U^n + 0.5*dt*dUdt^n
    MultiFab::LinComb(S_new, Real(0.5), Sborder, 0, Real(0.5), S_old, 0, 0, NCONS, 0);
    // S_new += 0.5*dt*dSdt
    MultiFab::Saxpy(S_new, Real(0.5)*dt, dSdt, 0, 0, NCONS, 0);
    // We now have S_new = U^{n+1} = (U^n+0.5*dt*dUdt^n) + 0.5*dt*dUdt^*

    return dt;
}


void CNS::compute_rhs (const MultiFab& statemf, MultiFab& dSdt, Real dt,
                   FluxRegister* fr_as_crse, FluxRegister* fr_as_fine)
{
  BL_PROFILE("CNS::compute_rhs()");
    
  //Aux Variables //////////////////////////////////////////////////////////////
  const auto dx     = geom.CellSizeArray();
  const auto dxinv  = geom.InvCellSizeArray();
  Parm const& lparm = *d_parm; // parameters (thermodynamic) -- TODO:add thermo namespace

  // numerical flux multifab array
  Array<MultiFab,AMREX_SPACEDIM> numflxmf; 
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      numflxmf[idim].define(statemf.boxArray(), statemf.DistributionMap(), NCONS, NGHOST); numflxmf[idim] = 0.0;}

  // primitive variables multifab
  MultiFab primsmf; primsmf.define(statemf.boxArray(), statemf.DistributionMap(), NPRIM, NGHOST);
  for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      auto const& statefab = statemf.array(mfi);
      auto const& prims    = primsmf.array(mfi);
      const Box& bxg       = mfi.growntilebox(NGHOST);

      amrex::ParallelFor(bxg,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      { cons2prim(i, j, k, statefab, prims, lparm);});
  }
  // ensure primitive variables mf computed before starting mfiter
  Gpu::streamSynchronize();
  //////////////////////////////////////////////////////////////////////////////


  //Euler Fluxes ///////////////////////////////////////////////////////////////
  // IMPROVEMENT: Can have pointer function (dynamic casting?) (main euler_flux function) which can be pointed to different flux schemes in the initialisation. The function can pass a parameter struct to include any scheme specfic parameters.
  // weno5js fvs
  if (euler_flux_type==2) {
    // make multifab for variables
    Array<MultiFab,AMREX_SPACEDIM> pntflxmf;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      pntflxmf[idim].define( statemf.boxArray(), statemf.DistributionMap(), NCONS, NGHOST); pntflxmf[idim] = 0.0; }

    FArrayBox lambdafab; //store eigenvalues max(u+c,u,u-c) in all directions
    // loop over all fabs
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // const Box& bx      = mfi.tilebox();
        const Box& bxg     = mfi.growntilebox(NGHOST);
        const Box& bxnodal = mfi.grownnodaltilebox(-1,0); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

        auto const& statefab = statemf.array(mfi);
        // auto const& dsdtfab = dSdt.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                     auto const& nfabfy = numflxmf[1].array(mfi);,
                     auto const& nfabfz = numflxmf[2].array(mfi););

        AMREX_D_TERM(auto const& pfabfx = pntflxmf[0].array(mfi);,
                     auto const& pfabfy = pntflxmf[1].array(mfi);,
                     auto const& pfabfz = pntflxmf[2].array(mfi););

        lambdafab.resize(bxg, 3);
        Elixir lambdagpu   = lambdafab.elixir();
        auto const& lambda = lambdafab.array();

        // x direction
        ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            cons2eulerflux_lambda(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lambda ,lparm);
        });

        // ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        //     cons2char(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lparm);
        // });

        ParallelFor(bxnodal, int(NCONS) , [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
              numericalflux_globallaxsplit(i, j, k, n, statefab ,pfabfx, pfabfy, pfabfz, lambda ,nfabfx, nfabfy, nfabfz); // Storage of numerical fluxes in nfabfx, nfabfy, nfabfz - index i contains i-1/2 interface flux.
        //       // printf("i,j,k,n  -  %i %i %i %i %f \n",i,j,k,n, nfabfx(i,j,k,n) );

        // y direction
        });
    }
  }
  
  // central-split KEEP + JST artificial dissipation
  else if (euler_flux_type==1) {
  // make multifab for spectral radius and sensor for artificial dissipation
    MultiFab lambdamf; lambdamf.define(statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);
    MultiFab sensormf; sensormf.define(statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);

    int order = 4;
    int halfsten = order/2;

    // Move this somewhere else
    //2 * standard finite difference coefficients
    GpuArray<Real,1> coeffs2; coeffs2[0]=Real(1.0);
    GpuArray<Real,2> coeffs4; coeffs4[0]=Real(4.0)/3 ,coeffs4[1]=Real(-2.0)/12;
    GpuArray<Real,3> coeffs6; coeffs6[0]=Real(6.0)/4; coeffs6[1]=Real(-6.0)/20;coeffs6[2]=Real(2.0)/60; 
    // elseif (halfCentOrder .eq. 8) then
    //    coeff(1:halfCentOrder) = (/ 4.d0/5.d0 , -1.d0/5.d0 , 4.d0/105.d0 , -1.d0/280.d0  /)

    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bx      = mfi.tilebox();
      const Box& bxnodal = mfi.grownnodaltilebox(-1,0);
      const Box& bxg     = mfi.growntilebox(NGHOST-1);

      auto const& statefab    = statemf.array(mfi);
      auto const& sensor   = sensormf.array(mfi);
      auto const& lambda   = lambdamf.array(mfi);
      auto const& prims    = primsmf.array(mfi);
      AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                    auto const& nfabfy = numflxmf[1].array(mfi);,
                    auto const& nfabfz = numflxmf[2].array(mfi););

      amrex::ParallelFor(bxnodal,  
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        KEEP(i,j,k,halfsten,coeffs4,prims,nfabfx,nfabfy,nfabfz,lparm);
      });


      // Order reduction at ymin wall boundary
      if (geom.Domain().smallEnd(1)==bx.smallEnd(1)) {
        int jbndry = bxnodal.smallEnd(1);
        IntVect small,big;
        small.setVal(0,bxnodal.smallEnd(0));
        small.setVal(1,jbndry);
        small.setVal(2,bxnodal.smallEnd(2));

        big.setVal(0,bxnodal.bigEnd(0));
        big.setVal(1,jbndry);
        big.setVal(2,bxnodal.bigEnd(2));
        Box bxboundary(small,big);

        amrex::ParallelFor(bxboundary, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          // printf("6th ord %f \n",nfabfy(i,j,k,2));
          KEEPy(i,j,k,1,coeffs2,prims,nfabfy,lparm);
          // printf("2nd ord %f \n",nfabfy(i,j,k,2));
          // KEEPy(i,j,k,2,coeffs4,prims,nfabfy,lparm);
          // printf("4th ord %f \n \n",nfabfy(i,j,k,2));
        });
      }

      // Order reduction at ymax wall boundary
      if (geom.Domain().bigEnd(1)==bx.bigEnd(1)) {
        int jbndry = bxnodal.bigEnd(1);
        IntVect small,big;
        small.setVal(0,bxnodal.smallEnd(0));
        small.setVal(1,jbndry);
        small.setVal(2,bxnodal.smallEnd(2));

        big.setVal(0,bxnodal.bigEnd(0));
        big.setVal(1,jbndry);
        big.setVal(2,bxnodal.bigEnd(2));
        Box bxboundary(small,big);

        amrex::ParallelFor(bxboundary, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          KEEPy(i,j,k,1,coeffs2,prims,nfabfy,lparm);
        });
      }


      // JST artificial dissipation shock capturing
     amrex::ParallelFor(bxg,
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
        // auto const& dsdtfab = dSdt.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                     auto const& nfabfy = numflxmf[1].array(mfi);,
                     auto const& nfabfz = numflxmf[2].array(mfi););

        const Box& bxg2 = amrex::grow(bx,2);
        qtmp.resize(bxg2, NPRIM);
        Elixir qeli = qtmp.elixir();
        auto const& q = qtmp.array();

        amrex::ParallelFor(bxg2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cons2prim(i, j, k, sfab, q, lparm);
        });

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
        qeli.clear(); // don't need them anymore
        slopeeli.clear();
    }
  }
  //////////////////////////////////////////////////////////////////////////////


  // Viscous Fluxes ////////////////////////////////////////////////////////////
  // Gpu::streamSynchronize(); // ensure all rhs terms computed before assembly
  // We have a separate MFIter loop here than the Euler fluxes and the source terms, so the work can be further parallised. As different MFIter loops can be in different GPU streams. 

  // Although conservative FD (finite difference) derivatives of viscous fluxes are not requried in the boundary layer, standard FD are likely sufficient. However, considering grid and flow discontinuities (coarse-interface flux-refluxing and viscous derivatives near shocks), conservative FD derivatives are preferred.

  // Make multifab for derivatives of primitive variables and viscous fluxes
  FArrayBox temp1;
  Array<MultiFab,AMREX_SPACEDIM> pntvflxmf;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    pntvflxmf[idim].define(statemf.boxArray(), statemf.DistributionMap(), NCONS, NGHOST); pntvflxmf[idim] = 0.0;}

  // loop over all fabs
  for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      // const Box& bxg  = mfi.growntilebox(NGHOST);
      const Box& bxpflx  = mfi.growntilebox(2);
      const Box& bxnodal  = mfi.grownnodaltilebox(-1,0); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

      // auto const& statefab = statemf.array(mfi);
      // auto const& dsdtfab  = dSdt.array(mfi);
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

  // Assemble RHS and add source terms /////////////////////////////////////////
  Gpu::streamSynchronize(); // ensure all rhs terms computed before assembly
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
}