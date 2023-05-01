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
    MultiFab Sborder(grids,dmap,NCONS,NUM_GROW,MFInfo(),Factory());

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
    FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NCONS);
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
    FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NCONS);

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

if (euler_flux_type==0) {
    const auto dx    = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = NCONS;
    const int neqns = NCONS;
    const int ncons = NCONS;
    const int nprim = NCONS+1;

    Array<MultiFab,AMREX_SPACEDIM> fluxes;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fluxes[idim].define(amrex::convert(statemf.boxArray(),IntVect::TheDimensionVector(idim)),
                            statemf.DistributionMap(), ncomp, 0);
        fluxes[idim] = 0.0;
    }

    Parm const* lparm = d_parm;

    FArrayBox qtmp, slopetmp;
    for (MFIter mfi(statemf); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        auto const& sfab = statemf.array(mfi);
        auto const& dsdtfab = dSdt.array(mfi);
        AMREX_D_TERM(auto const& fxfab = fluxes[0].array(mfi);,
                     auto const& fyfab = fluxes[1].array(mfi);,
                     auto const& fzfab = fluxes[2].array(mfi););

        const Box& bxg2 = amrex::grow(bx,2);
        qtmp.resize(bxg2, nprim);
        Elixir qeli = qtmp.elixir();
        auto const& q = qtmp.array();

        amrex::ParallelFor(bxg2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cons2prim(i, j, k, sfab, q, *lparm);
        });

        const Box& bxg1 = amrex::grow(bx,1);
        slopetmp.resize(bxg1,neqns);
        Elixir slopeeli = slopetmp.elixir();
        auto const& slope = slopetmp.array();

        // x-direction
        int cdir = 0;
        const Box& xslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(xslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_x(i, j, k, slope, q,*lparm);
        });
        const Box& xflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_x(i, j, k, fxfab, slope, q, *lparm);
            for (int n = neqns; n < ncons; ++n) fxfab(i,j,k,n) = Real(0.0);
        });

        // y-direction
        cdir = 1;
        const Box& yslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(yslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_y(i, j, k, slope, q, *lparm);
        });
        const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_y(i, j, k, fyfab, slope, q, *lparm);
            for (int n = neqns; n < ncons; ++n) fyfab(i,j,k,n) = Real(0.0);
        });

        // z-direction
        cdir = 2;
        const Box& zslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(zslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_z(i, j, k, slope, q, *lparm);
        });
        const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_z(i, j, k, fzfab, slope, q, *lparm);
            for (int n = neqns; n < ncons; ++n) fzfab(i,j,k,n) = Real(0.0);
        });

        // don't have to do this, but we could
        qeli.clear(); // don't need them anymore
        slopeeli.clear();

        amrex::ParallelFor(bx, ncons,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            cns_flux_to_dudt(i, j, k, n, dsdtfab, AMREX_D_DECL(fxfab,fyfab,fzfab), dxinv);
        });

        if (gravity != Real(0.0)) {
            const Real g = gravity;
            const int irho = Density;
            const int imz = Zmom;
            const int irhoE = Eden;
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dsdtfab(i,j,k,imz)   += g * sfab(i,j,k,irho);
                dsdtfab(i,j,k,irhoE) += g * sfab(i,j,k,imz);
            });
        }
    }

    if (fr_as_crse) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
            const Real scale = -dt*dA;
            fr_as_crse->CrseInit(fluxes[idim], idim, 0, 0, NCONS, scale, FluxRegister::ADD);
        }
    }

    if (fr_as_fine) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
            const Real scale = dt*dA;
            fr_as_fine->FineAdd(fluxes[idim], idim, 0, 0, NCONS, scale);
        }
    }

}

else {
    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();

    Array<MultiFab,AMREX_SPACEDIM> numflxmf;
    Array<MultiFab,AMREX_SPACEDIM> pntflxmf;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      numflxmf[idim].define(  amrex::convert(statemf.boxArray(),IntVect::TheDimensionVector(idim)), statemf.DistributionMap(), NCONS, NUM_GROW);
      pntflxmf[idim].define( amrex::convert(statemf.boxArray(),IntVect::TheDimensionVector(idim)), statemf.DistributionMap(), NCONS, NUM_GROW);
      numflxmf[idim] = 0.0;
      pntflxmf[idim] = 0.0;
    }

    Parm const* pparm = d_parm;

    // loop over all fabs
    for (MFIter mfi(statemf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx   = mfi.tilebox();
        const Box& bxg  = mfi.growntilebox(NUM_GROW);
        const Box& bxnodal  = mfi.grownnodaltilebox(-1,0); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

        auto const& statefab = statemf.array(mfi);
        auto const& dsdtfab = dSdt.array(mfi);
        AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);,
                     auto const& nfabfy = numflxmf[1].array(mfi);,
                     auto const& nfabfz = numflxmf[2].array(mfi););

        AMREX_D_TERM(auto const& pfabfx = pntflxmf[0].array(mfi);,
                     auto const& pfabfy = pntflxmf[1].array(mfi);,
                     auto const& pfabfz = pntflxmf[2].array(mfi););

        
        // Conservative to Primitive variable change ////////////////////
        // create primitive variable array
        // const Box& bxg = grow(bx,NUM_GROW);
        // qtmp.resize(bxg, NCONS);
        // Elixir qeli = qtmp.elixir();
        // auto const& qfab = qtmp.array();

        // amrex::ParallelFor(bxg,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     cns_ctoprim(i, j, k, qfab, q, *lparm);
        // });

        // Conservative to Characteristic variable change //////////////////////
        // const Box& bxg = grow(bx,NUM_GROW);
        // tempfab.resize(bxg, NCONS);
        // Elixir qeli = tempfab.elixir();
        // auto const& cfab = tempfab.array();
        // amrex::ParallelFor(bxg,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        //     constochar(i, j, k, statefab, charfab, 0);
        // });


        // Fluxes //////////////////////////////////////////////////////////////
        // Euler
        ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            cons2eulerflux(i, j, k, statefab, pfabfx, pfabfy, pfabfz, pparm);
        });

        ParallelFor(bxnodal, int(NCONS) , [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
              numericalflux(i, j, k, n, pfabfx, pfabfy, pfabfz ,nfabfx, nfabfy, nfabfz); // Storage of numerical fluxes in nfabfx, nfabfy, nfabfz - index i contains i-1/2 interface flux.
              // printf("i,j,k,n  -  %i %i %i %i %f \n",i,j,k,n, nfabfx(i,j,k,n) );
        });

        // Viscous terms
        // amrex::ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        //     cns_viscflux(i, j, k, cdir, q, w, 0);
        // });

        // add to rhs
        amrex::ParallelFor(bx, int(NCONS),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            cns_flux_to_dudt(i, j, k, n, dsdtfab, AMREX_D_DECL(nfabfx,nfabfy,nfabfz), dxinv);
        });

        // Source terms ////////////////////////////////////////////////////////

    }

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
}
}

// if (gravity != Real(0.0)) {
//     const Real g = gravity;
//     const int irho = Density;
//     const int imz = Zmom;
//     const int irhoE = Eden;
//     amrex::ParallelFor(bx,
//     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//     {
//         dsdtfab(i,j,k,imz) += g * statefab(i,j,k,irho);
//         dsdtfab(i,j,k,irhoE) += g * statefab(i,j,k,imz);
//     });
// }