
#include <AMReX_Vector.H>

namespace NLDE {
  // vector of baseflow multifabs
  inline Vector<MultiFab> Vbaseflow;


  // allocate baseflow multifabs
  inline void allocVMF(int& nlevs) {Vbaseflow.resize(nlevs);}


  // baseflow interpolation
  // current implementaiton is hard-coded analytical solution to supersonic shear layer
  inline void interpolate_baseflow() {
  
  MultiFab& basemf = Vbaseflow[lev];

   for (MFIter mfi(statemf, false); mfi.isValid(); ++mfi) {
      auto const& statefab = statemf.array(mfi);
      auto const& primsfab = primsmf.array(mfi);
      auto const& basefab  = basemf.array(mfi);
      const Box& bxg       = mfi.growntilebox(NGHOST);
      amrex::ParallelFor(bxg,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      { 
      // ***fill basefab(i,j,k,QRHO) = () --> indices for variables are in problem file
      // ***copied below from cns_prob.H 
      // ***need to check variables are available or include more header info

      //  const Real* prob_lo = geomdata.ProbLo();
      //  const Real* prob_hi = geomdata.ProbHi();
      //  const Real* dx      = geomdata.CellSize();      

      //  Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
      //  Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
      //  Real Pt, rhot, uxt, uyt, Tt, u1t, u2t, T1t, T2t;

      //  // mean pressure is uniform
      //  Pt = prob_parm.P1;

      //  // analytical expression for velocity distribution
      //  u1t = prob_parm.U1;
      //  u2t = prob_parm.U2;
      //  uxt = Real(0.5)*((u1t+u2t)+(u1t-u2t)*tanh(Real(2.0)*y/prob_parm.dw));
      
      //  // ananlytical expression for temperature distribution
      //  T1t = prob_parm.T1;
      //  T2t = prob_parm.T2;
      //  Tt = T1t*(T2t/T1t)*((1-(uxt/u1t))/(1-(u2t/u1t)))
      //      +T1t*(((uxt/u1t)-(u2t/u1t))/(1-(u2t/u1t)))
      //      +T1t*Real(0.5)*(closures.gamma-Real(1.0))*prob_parm.M1*prob_parm.M1*(1-(uxt/u1t))*((uxt/u1t)-(u2t/u1t));
  
      //  // density profile assuming perfect gas
      //  rhot = Pt/(closures.Rspec*Tt);
      
      //  state(i,j,k,URHO ) = rhot;
      //  state(i,j,k,UMX  ) = rhot*uxt;
      //  state(i,j,k,UMY  ) = Real(0.0);
      //  state(i,j,k,UMZ  ) = Real(0.0);
      //  Real et = Pt/(closures.gamma-Real(1.0));
      //  state(i,j,k,UET) = et + Real(0.5)*rhot*uxt*uxt;
  
      });
    }
  }


  inline void post_regrid(const int& lev,const BoxArray& grids, const DistributionMapping& dmap, const MFInfo& info, const FabFactory<FArrayBox>& factory) {
    // reallocate MF
    Vbaseflow[lev].clear();
    Vbaseflow[lev].define(grids,dmap,NCONS,0,info,factory);
    Vbaseflow[lev].setVal(0.0);

    // interpolate_baseflow();
  }


  // convert primitive variables to conserved variables
  inline void cons2prim(const int& lev, const MultiFab& statemf, MultiFab& primsmf ,const PROB::ProbClosures& cls) {

  MultiFab& basemf = Vbaseflow[lev];

   for (MFIter mfi(statemf, false); mfi.isValid(); ++mfi) {
      auto const& statefab = statemf.array(mfi);
      auto const& primsfab = primsmf.array(mfi);
      auto const& basefab  = basemf.array(mfi);
      const Box& bxg       = mfi.growntilebox(NGHOST);
      amrex::ParallelFor(bxg,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      { 
        Abort("NLDE cons2prim implement");
        // fill primsfab(i,j,k,QRHO) = () --> indices for variables are in problem file
        // TROUBLESHOOT: check manually at a point initialized state 
        
        // copied from CHAMPS: con2prim_NLDE subroutine in cons2prim.F90
        // subroutine cons2prim_NLDE(prim_bar,prim_dist,cons_bar,cons_dist,Ma_inf,gamma)
        //   !
        //   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //   !
        //   ! Description:
        //   ! Compute new flow field values with the new conservation values.
        //   ! For rho,u,v,w,T the direct equations and for p the equation of state is used.
        //   !
        //   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //   !
        //   use limiter_mod
        //   use typedef, only: pmin_int,Tmin_int,nvars_nse,nvars_Y,&
        //                     nvars_field,nl_z,mypeno,pmax_int,Tmax_int,&
        //                     icompu,icompv,icompw,icompt,icomptv,nTransportProperties,interp_props
        //   use material, only: Rgas,eos_T,eos_P
        // #if(MULTISPECIES==1)  
        //   use material_ms, only: Rgas_uni,mw_s,eos_rho_ms,eos_vib_temp,hf_s,isMolecule
        // #endif  
        //   !
        // #include "setup.h"
        //   !
        //   IMPLICIT NONE
        //   !
        //   REAL*8,INTENT(IN)            ::    prim_bar(nvars_nse+nTransportProperties)       ! p, u, v, (w,) T, (kappa,mu,Rgas,sos,gamma)
        //   REAL*8,INTENT(IN)            ::    cons_dist(nvars_nse)       !
        //   REAL*8,INTENT(IN)            ::    cons_bar(nvars_nse)
        //   REAL*8,INTENT(INOUT)         ::    Ma_inf,gamma
        //   REAL*8,INTENT(OUT)           ::    prim_dist(nvars_nse)       ! rho,rho*u,rho*v,rho*w,rho*(cv*T+0.5*KE)
        //   !
        //   ! local
        //   Real*8                       ::    invDens,R2U2,RhoU2,rho,nlz
        //   Integer                      ::    dir,n
        // #if(MULTISPECIES==1)
        //   REAL*8                       ::    cvr(1:nvars_Y),cvt(1:nvars_Y),rho_cv,prim_tot(1:nvars_nse)
        //   REAL*8                       ::    rho_cv_dist,hof_dist,rho_ms,rho_dist,rho_min
        //   REAL*8                       ::    inv_tot_den,Ev_total,Tv_total,Tv_tot_new
        // #endif  
        //   !
        //   ! u: prim_dist(UCOMP) = (cons_dist(RUCOMP) - cons_dist(RCOMP)*prim_bar(UCOMP))/invDens
        //   ! v: prim_dist(VCOMP) = (cons_dist(RVCOMP) - cons_dist(RCOMP)*prim_bar(VCOMP))/invDens
        //   ! w: prim_dist(WCOMP) = (cons_dist(RWCOMP) - cons_dist(RCOMP)*prim_bar(WCOMP))/invDens
        //   !
        //   ! 2D p:
        //   ! prim_dist(PCOMP) = -0.5d0*((gamma - 1.d0)*(cons_dist(RCOMP)*prim_bar(UCOMP)**2.d0 - &
        //   !      2.d0*cons_dist(ECOMP) + &
        //   !      cons_bar(RCOMP)*prim_dist(UCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_dist(UCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_bar(VCOMP)**2.d0 + &
        //   !      cons_bar(RCOMP)*prim_dist(VCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_dist(VCOMP)**2.d0 + &
        //   !      2.d0*cons_bar(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP) + &
        //   !      2.d0*cons_dist(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP) + &
        //   !      2.d0*cons_bar(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP) + &
        //   !      2.d0*cons_dist(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP)
        //   ! 3D p:
        //   ! prim_dist(PCOMP) = -0.5d0*((gamma - 1.d0)*(cons_dist(RCOMP)*prim_bar(UCOMP)**2.d0 - &
        //   !      2.d0*cons_dist(ECOMP) + &
        //   !      cons_bar(RCOMP)*prim_dist(UCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_dist(UCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_bar(VCOMP)**2.d0 + &
        //   !      cons_bar(RCOMP)*prim_dist(VCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_dist(VCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_bar(WCOMP)**2.d0 + &
        //   !      cons_bar(RCOMP)*prim_dist(WCOMP)**2.d0 + &
        //   !      cons_dist(RCOMP)*prim_dist(WCOMP)**2.d0 + &
        //   !      2.d0*cons_bar(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP) + &
        //   !      2.d0*cons_dist(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP) + &
        //   !      2.d0*cons_bar(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP) + &
        //   !      2.d0*cons_dist(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP) + &
        //   !      2.d0*cons_bar(RCOMP)*prim_bar(WCOMP)*prim_dist(WCOMP) + &
        //   !      2.d0*cons_dist(RCOMP)*prim_bar(WCOMP)*prim_dist(WCOMP)))
        //   !
        //   ! T: (P_p - R*T_0*rho_p)/(R*(rho_0 + rho_p))
        //   !
        //   if(interp_props) then
        //     gamma= prim_bar(TCOMP+5) !p,u,v,w,T,kappa,mu,Rgas,sos,gamma  
        //     Rgas = prim_bar(TCOMP+3)
        //     Ma_inf = 1.d0/sqrt(Rgas*gamma)
        //   endif
        //   !
        //   if ( nl_z == 1 )then
        //     nlz = 1.0d0
        //   elseif ( nl_z == 0 )then
        //     nlz = 0.d0
        //   else
        //     stop 'wrong value for nl_z'
        //   endif
        //   !
        // #if (MULTISPECIES==0)  
        //   invDens = 1.d0/(cons_bar(RCOMP) + cons_dist(RCOMP)*nlz)
        //   !
        //   prim_dist(UCOMP) = (cons_dist(RUCOMP) - cons_dist(RCOMP)*prim_bar(UCOMP))*invDens
        //   prim_dist(VCOMP) = (cons_dist(RVCOMP) - cons_dist(RCOMP)*prim_bar(VCOMP))*invDens
        //   !
        // #if (iins3d_axis3d==1 || iins3d_dist==1)
        //   !
        //   prim_dist(WCOMP) = (cons_dist(RWCOMP) - cons_dist(RCOMP)*prim_bar(WCOMP))*invDens
        //   !
        //   prim_dist(PCOMP) = -0.5d0*((gamma - 1.d0)*(cons_dist(RCOMP)*prim_bar(UCOMP)**2.d0 - &
        //       2.d0*cons_dist(ECOMP) + &
        //       cons_bar(RCOMP)*prim_dist(UCOMP)**2.d0*nlz  + &
        //       cons_dist(RCOMP)*prim_dist(UCOMP)**2.d0*nlz + &
        //       cons_dist(RCOMP)*prim_bar(VCOMP)**2.d0      + &
        //       cons_bar(RCOMP)*prim_dist(VCOMP)**2.d0*nlz  + &
        //       cons_dist(RCOMP)*prim_dist(VCOMP)**2.d0*nlz + &
        //       cons_dist(RCOMP)*prim_bar(WCOMP)**2.d0      + &
        //       cons_bar(RCOMP)*prim_dist(WCOMP)**2.d0*nlz  + &
        //       cons_dist(RCOMP)*prim_dist(WCOMP)**2.d0*nlz + &
        //       2.d0*cons_bar(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP)      + &
        //       2.d0*cons_dist(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP)*nlz + &
        //       2.d0*cons_bar(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP)      + &
        //       2.d0*cons_dist(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP)*nlz + &
        //       2.d0*cons_bar(RCOMP)*prim_bar(WCOMP)*prim_dist(WCOMP)      + &
        //       2.d0*cons_dist(RCOMP)*prim_bar(WCOMP)*prim_dist(WCOMP)*nlz   ))
        //   !
        // #else
        //   !
        //   prim_dist(PCOMP) = -0.5d0*((gamma - 1.d0)*(cons_dist(RCOMP)*prim_bar(UCOMP)**2.d0 - &
        //       2.d0*cons_dist(ECOMP) + &
        //       cons_bar(RCOMP)*prim_dist(UCOMP)**2.d0*nlz  + &
        //       cons_dist(RCOMP)*prim_dist(UCOMP)**2.d0*nlz + &
        //       cons_dist(RCOMP)*prim_bar(VCOMP)**2.d0      + &
        //       cons_bar(RCOMP)*prim_dist(VCOMP)**2.d0*nlz  + &
        //       cons_dist(RCOMP)*prim_dist(VCOMP)**2.d0*nlz + &
        //       2.d0*cons_bar(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP)      + &
        //       2.d0*cons_dist(RCOMP)*prim_bar(UCOMP)*prim_dist(UCOMP)*nlz + &
        //       2.d0*cons_bar(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP)      + &
        //       2.d0*cons_dist(RCOMP)*prim_bar(VCOMP)*prim_dist(VCOMP)*nlz ))
        //   !
        //   !
        // #endif
        //   !
        //   ! limiting total pressure, OB added Feb 2019
        //   prim_dist(PCOMP)=Max(prim_dist(PCOMP),-prim_bar(PCOMP)+pmin_int)
        //   ! need to limit max pressure as well because of computing flux in guards
        // !  prim_dist(PCOMP)=Min(prim_dist(PCOMP), prim_bar(PCOMP)-pmax_int)
        //   !
        //   prim_dist(TCOMP) = (prim_dist(PCOMP) - Rgas*prim_bar(TCOMP)*cons_dist(RCOMP))/( Rgas*(cons_bar(RCOMP) + cons_dist(RCOMP)*nlz ) )
        //   !
        //   ! limiting total temperature, OB added Feb 2019
        //   prim_dist(TCOMP)=Max(prim_dist(TCOMP),-prim_bar(TCOMP)+tmin_int)
        //   ! need to limit max temperature as well because of computing flux in guards
        // !  prim_dist(TCOMP)=Min(prim_dist(TCOMP), prim_bar(TCOMP)-tmax_int)
        //   !
        // #else

        //   do n=1,nvars_Y

        //       prim_dist(n) = cons_dist(n)                   !!! Species densities

        //       prim_tot(n)  = prim_bar(n) + cons_dist(n)

        //   end do

        //     rho_ms = eos_rho_ms(prim_bar(1:nvars_Y))                  !!! Mean total density

        //     rho_dist = eos_rho_ms(prim_dist(1:nvars_Y))               !!! Disturbance total density

        //     inv_tot_den = 1.d0/(rho_ms + rho_dist*dble(nlz))          !!! Inverse of total density

        //     prim_dist(icompu) = (cons_dist(icompu) - prim_bar(icompu)*rho_dist)*inv_tot_den   !!! Disturbance u

        //     prim_dist(icompv) = (cons_dist(icompv) - prim_bar(icompv)*rho_dist)*inv_tot_den   !!! Disturbance v

        // #if(iins3d_axis3d==1)

        //     prim_dist(icompw) = (cons_dist(icompw) - prim_bar(icompw)*rho_dist)*inv_tot_den   !!! Disturbance w

        // #endif  

        //     cvr         = 0.d0
        //     rho_cv      = 0.d0
        //     rho_cv_dist = 0.d0
        //     hof_dist    = 0.d0

        //   do n =1,nvars_Y
        //       if (isMolecule(n).gt.0) cvr(n) = Rgas_uni/mw_s(n)
        //   enddo

        //   do n =1, nvars_Y
        //       cvt(n) = 1.5*Rgas_uni/mw_s(n)
        //       rho_cv = rho_cv + prim_bar(n)*(cvt(n) + cvr(n))
        //       rho_cv_dist = rho_cv_dist + cons_dist(n)*(cvt(n) + cvr(n))
        //       hof_dist = hof_dist + cons_dist(n)*hf_s(n)
        //   enddo

        // !!! Translational temperature

        // #if(iins3d_axis3d==1)

        //     prim_dist(icompt) = (cons_dist(icompt) - rho_cv_dist*prim_bar(icompt) - 0.5*rho_ms*((prim_dist(icompu)**2 +  prim_dist(icompv)**2 + prim_dist(icompw)**2)*dble(nlz) + &

        //                         2.*prim_bar(icompu)*prim_dist(icompu) +  2.*prim_bar(icompv)*prim_dist(icompv) +  2.*prim_bar(icompw)*prim_dist(icompw)) - &

        //                         0.5*rho_dist*(prim_bar(icompu)**2 + prim_bar(icompv)**2 + prim_bar(icompw)**2 + &

        //                         (prim_dist(icompu)**2 + prim_dist(icompv)**2 + prim_dist(icompw)**2 + &

        //                         2.*prim_bar(icompu)*prim_dist(icompu) +  2.*prim_bar(icompv)*prim_dist(icompv) + 2.*prim_bar(icompw)*prim_dist(icompw))*dble(nlz)) - &

        //                         hof_dist - cons_dist(icomptv))/(rho_cv + rho_cv_dist*dble(nlz))

        // #else

        //     prim_dist(icompt) =  (cons_dist(icompt) - rho_cv_dist*prim_bar(icompt) - 0.5*rho_ms*((prim_dist(icompu)**2 + prim_dist(icompv)**2)*dble(nlz) + &

        //                         2.*prim_bar(icompu)*prim_dist(icompu) +  2.*prim_bar(icompv)*prim_dist(icompv)) - &

        //                         0.5*rho_dist*(prim_bar(icompu)**2 + prim_bar(icompv)**2 + &

        //                         (prim_dist(icompu)**2 + prim_dist(icompv)**2 + &

        //                         2.*prim_bar(icompu)*prim_dist(icompu) +  2.*prim_bar(icompv)*prim_dist(icompv))*dble(nlz)) - &

        //                         hof_dist - cons_dist(icomptv))/(rho_cv + rho_cv_dist*dble(nlz))
        // #endif



        //     Ev_total = cons_bar(icomptv) + cons_dist(icomptv)

        //     Tv_total = prim_bar(icomptv) + prim_dist(icomptv)

        //     call eos_vib_temp(prim_tot(1:nvars_nse),Ev_total,Tv_total,Tv_tot_new)

        //     prim_dist(icomptv) = Tv_tot_new - prim_bar(icomptv)    !!! Disturbance vibrational temperature

        //       rho_min = 1.e-20

        //       prim_dist(1:nvars_Y) = Max(prim_dist(1:nvars_Y),-prim_bar(1:nvars_Y)+ rho_min)

        //       prim_dist(icompt)    = Max(prim_dist(icompt),-prim_bar(icompt)+ Tmin_int)

        // #endif
        //   !
        // #if (DEBUG_NLDE==1)
        //   if (any(isnan(prim_dist(1:nvars_nse)))) then
        //     print*, "NaN in cons2prim_NLDE"
        //     print*, "prim_dist = ", prim_dist(1:nvars_nse)
        //     print*, "cons_dist = ", cons_dist(1:nvars_nse)
        //     print*, "BF prim = ", prim_bar(1:nvars_nse)
        //     print*, "BF cons = ", cons_bar(1:nvars_nse)
        //     print*, "mypeno = ", mypeno
        //     stop
        //   end if
        // #endif

        // end subroutine cons2prim_NLDE

      });
    }
  }


  // computes Euler fluxes of linear disturbances from conserved variable vector and maxeigen value
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
  void eflux_linear(int i, int j, int k, auto const& base,auto const& cons, auto const& fx, auto const& fy, auto const& fz, auto const& lambda ,const PROB::ProbClosures& cls) {

  Abort("NLDE Euler fluxes not implemented yet");

  // disturbance fluxes from eqn 15
  // also need baseflow quantities and fluxes
  //   Real rhoinv = Real(1.0)/cons(i,j,k,URHO);
  //   Real momx   = cons(i,j,k,UMX);
  //   Real momy   = cons(i,j,k,UMY);
  //   Real momz   = cons(i,j,k,UMZ);
  //   Real rhoet  = cons(i,j,k,UET);
  //   Real ux     = momx*rhoinv;
  //   Real uy     = momy*rhoinv;
  //   Real uz     = momz*rhoinv;

  //   Real rhoekin = Real(0.5)*rhoinv*(momx*momx + momy*momy + momz*momz);
  //   Real rhoeint = rhoet - rhoekin;
  //   Real P       = (closures.gamma - Real(1.0))*rhoeint;

  //   fx(i,j,k,URHO)  = momx;
  //   fx(i,j,k,UMX)   = momx*ux + P;
  //   fx(i,j,k,UMY)   = momy*ux;
  //   fx(i,j,k,UMZ)   = momz*ux;
  //   fx(i,j,k,UET)   = (rhoet + P)*ux;

  //   fy(i,j,k,URHO)  = momy;
  //   fy(i,j,k,UMX)   = momx*uy;
  //   fy(i,j,k,UMY)   = momy*uy + P;
  //   fy(i,j,k,UMZ)   = momz*uy;
  //   fy(i,j,k,UET)   = (rhoet + P)*uy;

  //   fz(i,j,k,URHO)  = momz;
  //   fz(i,j,k,UMX)   = momx*uz;
  //   fz(i,j,k,UMY)   = momy*uz;
  //   fz(i,j,k,UMZ)   = momz*uz + P;
  //   fz(i,j,k,UET)   = (rhoet + P)*uz;

  //   Real cs=sqrt(closures.gamma*P*rhoinv); 
  //   lambda(i,j,k,0) = std::abs(max(ux+cs,ux-cs,ux)); 
  //   lambda(i,j,k,1) = std::abs(max(uy+cs,uy-cs,uy)); 
  //   lambda(i,j,k,2) = std::abs(max(uz+cs,uz-cs,uz)); 
  }


  // euler flux computation function
  AMREX_FORCE_INLINE void eflux(const int& lev, const MultiFab& statemf, const MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {


  MultiFab& basemf = Vbaseflow[lev];

  // make multifab for variables
  Array<MultiFab,AMREX_SPACEDIM> pntflxmf;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    pntflxmf[idim].define( statemf.boxArray(), statemf.DistributionMap(), NCONS, NGHOST);
    }

  // store eigenvalues max(u+c,u,u-c) in all directions
  MultiFab lambdamf;
  lambdamf.define( statemf.boxArray(), statemf.DistributionMap(), AMREX_SPACEDIM, NGHOST);

  // loop over all fabs
  PROB::ProbClosures& lclosures = *CNS::d_prob_closures;
    for (MFIter mfi(statemf, false); mfi.isValid(); ++mfi)
    {
      const Box& bxg     = mfi.growntilebox(NGHOST);
      const Box& bxnodal = mfi.nodaltilebox(); // extent is 0,N_cell+1 in all directions -- -1 means for all directions. amrex::surroundingNodes(bx) does the same

      auto const& statefab = statemf.array(mfi);
      AMREX_D_TERM(auto const& nfabfx = numflxmf[0].array(mfi);, // nfabfx is numerical flux (i.e., computed fluxes at i,j,k cell cell centers)
                   auto const& nfabfy = numflxmf[1].array(mfi);,
                   auto const& nfabfz = numflxmf[2].array(mfi););

      AMREX_D_TERM(auto const& pfabfx = pntflxmf[0].array(mfi);, // pfabfx is physical flux (i.e., flux reconstruction at i+1/2, j+1/2, k+1/2 cell interfaces)
                   auto const& pfabfy = pntflxmf[1].array(mfi);,
                   auto const& pfabfz = pntflxmf[2].array(mfi););

      auto const& lambda = lambdamf.array(mfi);
      auto const& basefab = basemf.array(mfi);

      // if statement for linear disturance flux calculation (dist_linear)
      ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
          eflux_linear(i, j, k, basefab, statefab, pfabfx, pfabfy, pfabfz, lambda ,lclosures);
      });


      // transform to characteristic variables
      // ***makes things more accurate, especially near shocks, but not necessary for simulation to run***
      // ***output is pfabfx in transformed variables (pfabfx(cons) --> pfabfx(char))***
      // ParallelFor(bxg,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      //     cons2char(i, j, k, statefab, pfabfx, pfabfy, pfabfz, lclosures);
      // });


      // get fluxes at interfaces 
      // ***need this for simulation to run***
      // ***need to make dummy function as a placeholder for numerical flux (pfabfx --> nfabfx)***
      // ParallelFor(bxnodal, int(NCONS) , [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
      //       numericalflux_globallaxsplit(i, j, k, n, statefab ,pfabfx, pfabfy, pfabfz, lambda ,nfabfx, nfabfy, nfabfz); 
      // });
    }
  }


  // similarly, viscous fluxes
  AMREX_FORCE_INLINE void vflux(MultiFab& statemf,  MultiFab& primsmf, Array<MultiFab,AMREX_SPACEDIM>& numflxmf) {

  }


}