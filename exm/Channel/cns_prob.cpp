
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "CNS_index_macros.H"
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
  void amrex_probinit (const int* /*init*/, const int* /*name*/, const int* /*namelen*/, const amrex_real* /*problo*/, const amrex_real* /*probhi*/)
  {
    // amrex::ParmParse pp("prob");
    // pp.query("p_l", CNS::h_prob_parm->p_l);
    // pp.query("p_r", CNS::h_prob_parm->p_r);
    // pp.query("rho_l", CNS::h_prob_parm->rho_l);
    // pp.query("rho_r", CNS::h_prob_parm->rho_r);
    // pp.query("u_l", CNS::h_prob_parm->u_l);
    // pp.query("u_r", CNS::h_prob_parm->u_r);
#if AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#endif
  }

// void tagging(amrex::TagBoxArray& tags, amrex::MultiFab& sdata, int level){

// #ifdef AMREX_USE_OMP
// #pragma omp parallel if (Gpu::notInLaunchRegion())
// #endif
//   Real dengrad_threshold= Real(-1.2);
//   for (amrex::MFIter mfi(tags,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//   {
//     const amrex::Box& bx = mfi.tilebox();
//     auto const& tagfab = tags.array(mfi);
//     auto const& sdf = sdata.array(mfi); // state data fab (sdf)
//     int idx = -1; // density index
//     amrex::ParallelFor(bx,
//     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept 
//     {
//       amrex::Real ax = amrex::Math::abs(sdf(i+0,j,k,idx) - sdf(i,j,k,idx));
//       amrex::Real ay = amrex::Math::abs(sdf(i,j+0,k,idx) - sdf(i,j,k,idx));
//       amrex::Real az = amrex::Math::abs(sdf(i,j,k+0,idx) - sdf(i,j,k,idx));
//       ax = amrex::max(ax,amrex::Math::abs(sdf(i,j,k,idx) - sdf(i-2,j,k,idx)));
//       ay = amrex::max(ay,amrex::Math::abs(sdf(i,j,k,idx) - sdf(i,j-2,k,idx)));
//       az = amrex::max(az,amrex::Math::abs(sdf(i,j,k,idx) - sdf(i,j,k-2,idx)));
//       if (amrex::max(ax,ay,az) >= dengrad_threshold) {
//           tagfab(i,j,k) = true;}
//     });
//   }
            // amrex::Print() << "i,j,k         " << i << " " << j << " " << k << " "<< std::endl;
            // amrex::Print() << "tag(i,j,k)    " << int(tagfab(i,j,k)) << std::endl;
            // amrex::Print() << "data(i,j,k,0) " << datafab(i,j,k,0) << std::endl;
            // amrex::Print() << "drhox         " << drhox << std::endl;
            // amrex::Print() << "------------- " << std::endl;


// AMREX_GPU_HOST_DEVICE
// inline
// void
// cns_tag_denerror (int i, int j, int k,
//                   amrex::Array4<char> const& tag,
//                   amrex::Array4<amrex::Real const> const& rho,
//                   amrex::Real dengrad_threshold, char tagval) noexcept
// {
//     amrex::Real ax = amrex::Math::abs(rho(i+1,j,k) - rho(i,j,k));
//     amrex::Real ay = amrex::Math::abs(rho(i,j+1,k) - rho(i,j,k));
//     amrex::Real az = amrex::Math::abs(rho(i,j,k+1) - rho(i,j,k));
//     ax = amrex::max(ax,amrex::Math::abs(rho(i,j,k) - rho(i-1,j,k)));
//     ay = amrex::max(ay,amrex::Math::abs(rho(i,j,k) - rho(i,j-1,k)));
//     az = amrex::max(az,amrex::Math::abs(rho(i,j,k) - rho(i,j,k-1)));
//     if (amrex::max(ax,ay,az) >= dengrad_threshold) {
//         tag(i,j,k) = tagval;
//     }
// }

}
