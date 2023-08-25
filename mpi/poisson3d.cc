#include <mpi.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <algorithm>

void CopySendBuf(double ****phi, int iStart, int iEnd,
                  int jStart, int jEnd, 
                  int kStart, int kEnd, 
                  int disp, int dir, double* fieldSend, int MaxBufLen);

void CopyRecvBuf(double ****phi, int iStart, int iEnd, 
                  int jStart, int jEnd, 
                  int kStart, int kEnd, 
                  int disp, int dir, double* fieldRecv, int MaxBufLen);

void Jacobi_sweep(int nx, int ny, int nz, double ****phi, int t0, int t1,
      int **udim, double h, double* maxdelta);

int main(int argc, char* argv[])
{
    int pbc_check[3];
    int spat_dim[3], proc_dim[3], loca_dim[3], mycoord[3], totmsgsize[3];
		int i, myid, numprocs, ierr, itermax, tag;
		bool l_reorder;
		int myid_grid, nump_grid, tmp, t0, t1;
		
		MPI_Comm GRID_COMM_WORLD;
		MPI_Request req;
		MPI_Status status;
		
	  int iStart, jStart, kStart, iEnd, jEnd, kEnd, MaxBufLen;
	  int source, dest, dir, disp, iter;
	  
	  int **udim;
	  udim = new int*[2];
	  udim[0] = new int[3];
	  udim[1] = new int[3];
	  
	  double eps, maxdelta, h;
	  double (****phi), *fieldSend, *fieldRecv;

		MPI_Init(&argc, &argv);
		ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
		
		// TODO read parameters from file
		if (myid == 0)
		{
		  tmp = 1000;
		  
		  proc_dim[0] = 2;
		  proc_dim[1] = 2;
		  proc_dim[2] = 1;
		  
		  itermax = 10000;
		  
		  eps = 1e-10;
		  
		  if (numprocs != proc_dim[0] * proc_dim[1]*proc_dim[2])
		  {
		    std::cout << "Total procs cannot to factorized\n"
		    << "Total procs = " << numprocs << '\n'
		    << "Proc grid   = " << proc_dim[0] << proc_dim[1] << proc_dim[2] << '\n';
		    
		    ierr = MPI_Abort(MPI_COMM_WORLD, tmp);
		  }
		  
		  int spat_dim[] = {tmp, tmp, tmp};
		  int pbc_check[] = {false, false, false};
		}
		
		ierr = MPI_Bcast(spat_dim, 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
		ierr = MPI_Bcast(proc_dim, 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
		ierr = MPI_Bcast(pbc_check,3, MPI_LOGICAL, 0, MPI_COMM_WORLD);
		ierr = MPI_Bcast(&itermax, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
		ierr = MPI_Bcast(&eps,     1, MPI_DOUBLE,  0, MPI_COMM_WORLD);
		
		ierr = MPI_Dims_create(numprocs, 3, proc_dim);
		
		h = 1./(spat_dim[0]-1);
		
		if (myid == 0) {
		  std::cout << "Spatial Grid: " << spat_dim[0] << spat_dim[1] << spat_dim[2];
		  std::cout << "MPI     Grid: " << proc_dim[0] << proc_dim[1] << proc_dim[2];
		  std::cout << "Spatial h   : " << h;
		  std::cout << "itermax     : " << itermax;
		  std::cout << "eps         : " << eps;
		}
		
		l_reorder = true;

    // create process topology
		ierr = MPI_Cart_create(MPI_COMM_WORLD, 3, proc_dim, pbc_check, l_reorder, &GRID_COMM_WORLD);
		
		if (GRID_COMM_WORLD == MPI_COMM_NULL)
		{
		  if (myid == 0)
		    std::cout << "Failed to create GRID_COMM_WORLD\n";
		  ierr = MPI_Abort(MPI_COMM_WORLD, tmp);
		}
		
		// get rank and size for this topology
		ierr = MPI_Comm_rank(GRID_COMM_WORLD, &myid_grid);
		ierr = MPI_Comm_size(GRID_COMM_WORLD, &nump_grid);
		
		// get grid coordinates for this rank
		ierr = MPI_Cart_coords(GRID_COMM_WORLD, myid_grid, 3, mycoord);
		
		// loca_dim = grid size owned by current rank
		for (int i = 0; i < 3; ++i)
		{
		  loca_dim[i] = spat_dim[i] / proc_dim[i];

		  // TODO Don't understand this
		  if (mycoord[i] < spat_dim[i] % proc_dim[i])
		    loca_dim[i] += 1;
		}
		
		// Solution variables
		// One layer of ghost points on all sides
		
		iStart = 0; iEnd = loca_dim[2] + 1;
		jStart = 0; jEnd = loca_dim[1] + 1;
		kStart = 0; kEnd = loca_dim[0] + 1;
		
		phi = new double***[iEnd + 1 - iStart];
  	for (int i = 0; i < iEnd - iStart; ++i)
    {
      phi[i] = new double**[jEnd + 1 - jStart];
      for (int j = 0; j < jEnd - jStart; ++j)
      {
        phi[i][j] = new double*[kEnd + 1 - kStart];
        for (int k = 0; k < kEnd - kStart; ++k)
          phi[i][j][k] = new double[2];
      }
    }
		
		MaxBufLen = 0;
		
		totmsgsize[2] = loca_dim[0] * loca_dim[1];
		MaxBufLen = std::max(MaxBufLen, totmsgsize[2]);

		totmsgsize[1] = loca_dim[0] * loca_dim[2];
		MaxBufLen = std::max(MaxBufLen, totmsgsize[1]);

		totmsgsize[0] = loca_dim[1] * loca_dim[2];
		MaxBufLen = std::max(MaxBufLen, totmsgsize[0]);

    fieldSend = new double[MaxBufLen];
    fieldRecv = new double[MaxBufLen];
    
    disp = -1;
    
    for (int dir = 0; dir < 3; ++dir)
    {
      ierr = MPI_Cart_shift(GRID_COMM_WORLD, dir, disp, &source, &dest);
      if (dest != MPI_PROC_NULL)
        udim[0][dir] = 1;
      else
        udim[0][dir] = 2;
      if (source != MPI_PROC_NULL)
        udim[1][dir] = loca_dim[dir];
      else
        udim[1][dir] = loca_dim[dir]-1;
    }

    // TODO
    // phi = 0.;
    
    // Begin iterations
    maxdelta = 2. * eps;
    t0 = 0; t1 = 1;
    tag = 0;
    iter = 0;
    
    while(iter < itermax && maxdelta > eps)
    {
      for (int disp : {-1, 1}){
        for (int dir = 0; dir < 3; ++dir)
        {
          MPI_Cart_shift(GRID_COMM_WORLD, dir, disp, &source, &dest);

          if (source != MPI_PROC_NULL)
            MPI_Irecv(&fieldRecv[0], totmsgsize[dir], MPI_DOUBLE_PRECISION, source, tag, GRID_COMM_WORLD, &req);
            
          if (dest != MPI_PROC_NULL)
          {
            CopySendBuf(phi, iStart, iEnd, jStart, jEnd, kStart, kEnd,
            disp, dir, fieldSend, MaxBufLen);
            MPI_Send(fieldSend, totmsgsize[dir], MPI_DOUBLE_PRECISION, dest, tag, GRID_COMM_WORLD);
          }
          
          if (source != MPI_PROC_NULL)
          {
            MPI_Wait(&req, &status);
            CopyRecvBuf(phi, iStart, iEnd, jStart, jEnd, kStart, kEnd,
                        disp, dir, fieldRecv, MaxBufLen);
          }
        }
        }
      Jacobi_sweep(loca_dim[2], loca_dim[1], loca_dim[0],
      phi, t0, t1, udim, h, &maxdelta);
      
      MPI_Allreduce(MPI_IN_PLACE, &maxdelta, 1, MPI_DOUBLE_PRECISION, MPI_MAX, GRID_COMM_WORLD);
      
      iter += 1;
      if (myid == 1) {
        std::cout << iter << ", " << maxdelta;
        tmp = t0; t0 = t1; t1 = tmp;
      }
    }
    
		ierr = MPI_Finalize();
		return ierr;
}

void CopySendBuf(double ****phi, int iStart, int iEnd, 
                  int jStart, int jEnd, 
                  int kStart, int kEnd, 
                  int disp, int direction, double* fieldSend, int MaxBufLen)
{
  int i1, i2, j1, j2, k1, k2, c;
  int i,j,k;
  
  if (direction < 1 || direction > 3)
  {
    std::cout << "CSB: dir is wrong\n";
    exit(1);
  }
  if (disp != 1 || disp != -1)
  {
    std::cout << "CSB: disp is wrong\n";
    exit(1);
  }
  
  if (direction == 0)
  {
    i1 = iStart + 1; i2 = iEnd - 1;
    j1 = jStart + 1; j2 = jEnd - 1;
    
    if (disp == -1)
      k1 = k2 = 1;
    else
      k1 = k2 = kEnd - 1;
  }
  else if(direction == 1)
  {
    i1 = iStart + 1; i2 = iEnd - 1;
    k1 = kStart + 1; k2 = kEnd - 1;
    
    if (disp == -1)
      j1 = j2 = 1;
    else
      j1 = j2 = jEnd - 1;
  }
  else if(direction == 2)
  {
    j1 = jStart + 1; j2 = jEnd - 1;
    k1 = kStart + 1; k2 = kEnd - 1;
    
    if (disp == -1)
      i1 = i2 = 1;
    else
      i1 = i2 = jEnd - 1;
  }
  
  c = 1;
  for (int k = k1; k < k2; ++k)
    for (int j = j1; j < j2; ++j)
      for (int i = i1; i < i2; ++i)
      {
        fieldSend[c] = phi[i][j][k][0];
        c += 1;
      }
  return;
}

void CopyRecvBuf(double ****phi, int iStart, int iEnd, 
                  int jStart, int jEnd, 
                  int kStart, int kEnd, 
                  int disp, int dir, double* fieldRecv, int MaxBufLen)
{
  int i1, i2, j1, j2, k1, k2, c;
  int i,j,k;
  
  if (dir < 1 || dir > 3)
  {
    std::cout << "CRB: dir is wrong\n";
    exit(1);
  }
  if (disp != 1 || disp != -1)
  {
    std::cout << "CRB: disp is wrong\n";
    exit(1);
  }
  
  if (dir == 0)
  {
    i1 = iStart + 1; i2 = iEnd - 1;
    j1 = jStart + 1; j2 = jEnd - 1;
    
    if (disp == 1)
      k1 = k2 = 0;
    else
      k1 = k2 = kEnd;
  }
  else if(dir == 1)
  {
    i1 = iStart + 1; i2 = iEnd - 1;
    k1 = kStart + 1; k2 = kEnd - 1;
    
    if (disp == 1)
      j1 = j2 = 0;
    else
      j1 = j2 = jEnd;
  }
  else if(dir == 2)
  {
    j1 = jStart + 1; j2 = jEnd - 1;
    k1 = kStart + 1; k2 = kEnd - 1;
    
    if (disp == 1)
      i1 = i2 = 0;
    else
      i1 = i2 = jEnd;
  }
  
  c = 1;
  for (int k = k1; k < k2; ++k)
    for (int j = j1; j < j2; ++j)
      for (int i = i1; i < i2; ++i)
      {
        phi[i][j][k][0] = fieldRecv[c];
        c += 1;
      }
  return;
}

void Jacobi_sweep(int nx, int ny, int nz, double**** phi, int t0, int t1,
int** udim, double h, double* maxdelta)
{
		double rhs = 1.0;
		double one_over_six = 1./6.;

  int i, j, k;
  *maxdelta = 0.;
  
  for (int k = udim[0][0]; k < udim[1][0]; ++k)
    for(int j = udim[0][1]; j < udim[1][1]; ++j)
      for (int i = udim[0][2]; i < udim[1][2]; ++i)
      {
        phi[i][j][k][t1] = 
        ( phi[i-1][j][k][t0] + phi[i+1][j][k][t0]
        + phi[i][j-1][k][t0] + phi[i][j+1][k][t0]
        + phi[i][j][k-1][t0] + phi[i][j][k+1][t0]
        + h*h * rhs) * one_over_six;
        *maxdelta = std::max(*maxdelta, abs(phi[i][j][k][t1] - phi[i][j][k][t0]));
      }
  return;
}
