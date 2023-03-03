// #include <CNS.H>
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include <cns_prob.H>

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        // could read parmparse parameters here

        // amrex::Gpu::htod_memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));

        // tagging
    }
}


void cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real Pt, rhot, uxt;
    if (x < Real(0.5)*(prob_lo[0]+prob_hi[0])) {
        Pt   = prob_parm.p_l;
        rhot = prob_parm.rho_l;
        uxt  = prob_parm.u_l;
    } else {
        Pt   = prob_parm.p_r;
        rhot = prob_parm.rho_r;
        uxt  = prob_parm.u_r;
    }
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot*uxt;
    state(i,j,k,UMY  ) = Real(0.0);
    state(i,j,k,UMZ  ) = Real(0.0);
    Real eint = Pt/(parm.eos_gamma-Real(1.0));
    state(i,j,k,UEINT) = eint;
    state(i,j,k,UEDEN) = eint + Real(0.5)*rhot*uxt*uxt;
    state(i,j,k,UTEMP) = Real(0.0);

    // amrex::Print() << i << " " << j << " " << k << " " << "\n";
    // amrex::Print() << prob_parm.p_r << "  "<< parm.eos_gamma << "\n";
    // exit(0);
}


void tagging(amrex::TagBoxArray& tags, amrex::MultiFab& data, int level){

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(tags,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& tagfab = tags.array(mfi);
        auto const& datafab = data.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          // Temporary tagging on density in x only
          amrex::Real drhox = amrex::Math::abs(datafab(i+1,j,k,0) - datafab(i-1,j,k,0))/datafab(i,j,k,0);
          tagfab(i,j,k) = drhox > 0.5f;

          // amrex::Print() << "i,j,k         " << i << " " << j << " " << k << " "<< std::endl;
          // amrex::Print() << "tag(i,j,k)    " << int(tagfab(i,j,k)) << std::endl;
          // amrex::Print() << "data(i,j,k,0) " << datafab(i,j,k,0) << std::endl;
          // amrex::Print() << "drhox         " << drhox << std::endl;
          // amrex::Print() << "------------- " << std::endl;
        });
    }
}
