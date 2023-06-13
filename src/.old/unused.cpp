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

