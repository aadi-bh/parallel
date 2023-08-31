#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <string>

void CopySendBuf(double ****phi, int t, int iStart, int iEnd, int jStart,
                 int jEnd, int kStart, int kEnd, int disp, int dir,
                 double *fieldSend, int MaxBufLen);

void CopyRecvBuf(double ****phi, int t, int iStart, int iEnd, int jStart,
                 int jEnd, int kStart, int kEnd, int disp, int dir,
                 double *fieldRecv, int MaxBufLen);

void Jacobi_sweep(int nx, int ny, int nz, double ****phi, int t0, int t1,
                  int **udim, double h, double &maxdelta);
void write_rectilinear_grid(int id, int nx, int ny, int nz, double ****var,
                            double t, int c);

int main(int argc, char *argv[]) {
  // Mark whether boundaries are periodic or not
  int pbc_check[3];
  // Number of cells in each dimension
  int spat_dim[3];
  // Number of processes in each dimension
  int proc_dim[3];
  // TODO what's loca_dim?
  int loca_dim[3];
  // TODO I have a guess for what's mycoord
  int mycoord[3];
  int totmsgsize[3];
  int i, myid, numprocs, ierr, itermax, tag;
  int myid_grid, nump_grid, tmp, t0, t1;

  MPI_Comm GRID_COMM_WORLD;
  MPI_Request req;
  MPI_Status status;

  int iStart, jStart, kStart, iEnd, jEnd, kEnd, MaxBufLen;
  int source, dest, dir, displacement;

  // A 2-d array for storing the start and end indices to
  // apply the sweep on.
  // Indexed by [direction][dimension]
  int **udim;
  udim = new int *[2];
  udim[0] = new int[3];
  udim[1] = new int[3];

  double eps, maxdelta, h;
  double(****phi), *fieldSend, *fieldRecv;

  // Initialise MPI
  //  MPI::Init(&argc, &argv);
  MPI_Init(&argc, &argv);

  // Set numprocs to be the number of ranks, which is set at run time.
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  // Each rank has an id. Sets myid to this particular process' id.
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0) {
    // TODO Don't hardcode these values
    tmp = 10;

    // Number of ranks for each dimension
    proc_dim[0] = 2;
    proc_dim[1] = 1;
    proc_dim[2] = 1;

    itermax = 10000;

    eps = 1e-10;

    // Total number of processes = product of the number of processes along each
    // dimension.
    if (numprocs != proc_dim[0] * proc_dim[1] * proc_dim[2]) {
      std::cout << "Total procs cannot to factorized\n"
                << "Total procs = " << numprocs << '\n'
                << "Proc grid   = " << proc_dim[0] << proc_dim[1] << proc_dim[2]
                << '\n';

      ierr = MPI_Abort(MPI_COMM_WORLD, tmp);
    }

    // Number of cells along each dimension.
    spat_dim[0] = spat_dim[1] = spat_dim[2] = tmp;

    // Don't treat boundaries as periodic along any dimension
    pbc_check[0] = pbc_check[1] = pbc_check[2] = false;
  }

  // Bcast sends the root process' value to all other processes.
  ierr = MPI_Bcast(spat_dim, 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
  ierr = MPI_Bcast(proc_dim, 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
  ierr = MPI_Bcast(pbc_check, 3, MPI_LOGICAL, 0, MPI_COMM_WORLD);
  ierr = MPI_Bcast(&itermax, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  ierr = MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Partition ranks into a 3D topology, with proc_dim ranks along each dim
  ierr = MPI_Dims_create(numprocs, 3, proc_dim);

  // TODO assuming dx = dy = dz, so no need for separate hx, hy, hz
  // Minus 1 because The last cell will be at the end
  h = 1. / (spat_dim[0] - 1);

  if (myid == 0) {
    std::cout << "Spatial Grid: " << spat_dim[0] << ' ' << spat_dim[1] << ' '
              << spat_dim[2] << '\n';
    std::cout << "MPI     Grid: " << proc_dim[0] << ' ' << proc_dim[1] << ' '
              << proc_dim[2] << '\n';
    std::cout << "Spatial h   : " << h << '\n';
    std::cout << "itermax     : " << itermax << '\n';
    std::cout << "eps         : " << eps << std::endl;
  }

  bool reorder = true;
  // create new communicator GRID_COMM_WORLD which has the topology info
  // reorder=false would restrict it to keep old and new ids the same,
  // but reorder=true allows us to give that up in return for possibly
  // better numbering.
  ierr = MPI_Cart_create(MPI_COMM_WORLD, 3, proc_dim, pbc_check, reorder,
                         &GRID_COMM_WORLD);

  if (GRID_COMM_WORLD == MPI_COMM_NULL) {
    if (myid == 0)
      std::cout << "Failed to create GRID_COMM_WORLD\n";
    MPI_Abort(MPI_COMM_WORLD, tmp);
  }

  // get size and rank from this new communicator
  ierr = MPI_Comm_size(GRID_COMM_WORLD, &nump_grid);
  ierr = MPI_Comm_rank(GRID_COMM_WORLD, &myid_grid);

  // get grid coordinates for this rank
  ierr = MPI_Cart_coords(GRID_COMM_WORLD, myid_grid, 3, mycoord);

  // loca_dim is the grid size assigned to each current rank
  for (int i = 0; i < 3; ++i) {
    // Number of cells along i'th dimension divided by number of ranks along
    // that dimension This is truncated division, a/b = a//b * b + a%b
    loca_dim[i] = spat_dim[i] / proc_dim[i];

    // Compensate for the truncation.
    // Increase the size of (a%b) by one.
    if (mycoord[i] < spat_dim[i] % proc_dim[i])
      loca_dim[i] += 1;
  }

  // Solution variables
  // One layer of ghost points on all sides
  iStart = 0;
  iEnd = loca_dim[2] + 1;
  jStart = 0;
  jEnd = loca_dim[1] + 1;
  kStart = 0;
  kEnd = loca_dim[0] + 1;

  // Initialise the solution array: phi[i][j][k][t] with zeroes
  // where t = 0 and 1 so that we can hold one older solution as well
  phi = new double ***[iEnd + 1 - iStart];
  for (int i = 0; i < iEnd - iStart; ++i) {
    phi[i] = new double **[jEnd + 1 - jStart];
    for (int j = 0; j < jEnd - jStart; ++j) {
      phi[i][j] = new double *[kEnd + 1 - kStart];
      for (int k = 0; k < kEnd - kStart; ++k) {
        phi[i][j][k] = new double[2];
        phi[i][j][k][0] = 0.;
        phi[i][j][k][1] = 0.;
      }
    }
  }

  MaxBufLen = 0;

  totmsgsize[2] = loca_dim[0] * loca_dim[1];
  MaxBufLen = std::max(MaxBufLen, totmsgsize[2]);

  totmsgsize[1] = loca_dim[0] * loca_dim[2];
  MaxBufLen = std::max(MaxBufLen, totmsgsize[1]);

  totmsgsize[0] = loca_dim[1] * loca_dim[2];
  MaxBufLen = std::max(MaxBufLen, totmsgsize[0]);

  // Buffers to send and receive data
  fieldSend = new double[MaxBufLen];
  fieldRecv = new double[MaxBufLen];

  displacement = -1;

  // udim is different for each rank, and stores which points to update
  // and which ones to ignore because they will be filled in Dirichlet BC.
  for (int direction = 0; direction < 3; ++direction) {
    // We want to transfer some data to the neighbouring ranks.
    // Cart_shift with the direction and displacement tells us which rank to
    // send the data to and which rank to receive from, depending on the
    // topology.
    ierr = MPI_Cart_shift(GRID_COMM_WORLD, direction, displacement, &source,
                          &dest);
    // If boundary isn't periodic, then shifting out of range will be signified
    // by NULL
    if (dest != MPI_PROC_NULL)
      // We have a neighbour on the left
      // So update every cell, starting from 0
      udim[0][direction] = 1;
    else
      // When no neighbour on the left, Dirichlet BC, no need to update the 0th
      udim[0][direction] = 2;
    if (source != MPI_PROC_NULL)
      // Neighbour on the left, so index is the last one
      udim[1][direction] = loca_dim[direction];
    else
      // No neighbour, Dirichlet BC, so no need to update this
      udim[1][direction] = loca_dim[direction] - 1;
  }

  // Begin iterations
  maxdelta = 2. * eps;

  // just indices for phi
  t0 = 0;
  t1 = 1;

  // tags are optional, meant so that ranks can easily differentiate between
  // messages
  tag = 0;

  for (int iter = 0; iter < itermax && maxdelta > eps; ++iter) {
    for (int displacement : {-1, 1}) {
      for (int direction = 0; direction < 3; ++direction) {
        MPI_Cart_shift(GRID_COMM_WORLD, direction, displacement, &source,
                       &dest);

        if (source != MPI_PROC_NULL)
          // We have a neighbour, so we receive into fieldRecv, the total
          // msgsize amount of data in that given direction. Irecv so that
          // everybody is ready to receive before everybody sends
          // TODO Replace this with SendRecv_replace?
          MPI_Irecv(&fieldRecv[0], totmsgsize[direction], MPI_DOUBLE_PRECISION,
                    source, tag, GRID_COMM_WORLD, &req);

        if (dest != MPI_PROC_NULL) {
          // Copy into the send buffer the data to send, which is phi_old
          CopySendBuf(phi, t0, iStart, iEnd, jStart, jEnd, kStart, kEnd,
                      displacement, direction, fieldSend, MaxBufLen);

          // then send that buffer to dest.
          MPI_Send(fieldSend, totmsgsize[direction], MPI_DOUBLE_PRECISION, dest,
                   tag, GRID_COMM_WORLD);
        }

        // Consequences of the Irecv
        if (source != MPI_PROC_NULL) {
          MPI_Wait(&req, &status);
          // Copy data from the receive buffer into phi_old
          CopyRecvBuf(phi, t0, iStart, iEnd, jStart, jEnd, kStart, kEnd,
                      displacement, direction, fieldRecv, MaxBufLen);
        }
      }
    }
    // All communication done, we have the latest values. Now compute new values
    Jacobi_sweep(loca_dim[2], loca_dim[1], loca_dim[0], phi, t0, t1, udim, h,
                 maxdelta);

    // Find the largest delta amongst all ranks, and set it to the max_delta in
    // every rank
    MPI_Allreduce(MPI_IN_PLACE, &maxdelta, 1, MPI_DOUBLE_PRECISION, MPI_MAX,
                  GRID_COMM_WORLD);

    if (myid == 0) {
      std::cout << std::fixed;
      std::cout << iter << ", " << maxdelta << std::endl;

      // New becomes old
      std::swap(t0, t1);
    }
  }

  ierr = MPI_Finalize();

  write_rectilinear_grid(myid_grid, loca_dim[2], loca_dim[1], loca_dim[0], phi,
                         0., 0);
  return ierr;
}

void CopySendBuf(double ****phi, int t, int iStart, int iEnd, int jStart,
                 int jEnd, int kStart, int kEnd, int disp, int direction,
                 double *fieldSend, int MaxBufLen) {
  int i1, i2, j1, j2, k1, k2, c;
  int i, j, k;

  if (direction < 0 || direction > 2) {
    std::cout << "CSB: dir is wrong\n";
    exit(1);
  }
  if (disp != 1 && disp != -1) {
    std::cout << "CSB: disp is wrong\n";
    exit(1);
  }

  if (direction == 0) {
    i1 = iStart + 1;
    i2 = iEnd - 1;
    j1 = jStart + 1;
    j2 = jEnd - 1;

    if (disp == -1)
      k1 = k2 = 1;
    else
      k1 = k2 = kEnd - 1;
  } else if (direction == 1) {
    i1 = iStart + 1;
    i2 = iEnd - 1;
    k1 = kStart + 1;
    k2 = kEnd - 1;

    if (disp == -1)
      j1 = j2 = 1;
    else
      j1 = j2 = jEnd - 1;
  } else if (direction == 2) {
    j1 = jStart + 1;
    j2 = jEnd - 1;
    k1 = kStart + 1;
    k2 = kEnd - 1;

    if (disp == -1)
      i1 = i2 = 1;
    else
      i1 = i2 = jEnd - 1;
  }

  c = 1;
  for (int k = k1; k < k2; ++k)
    for (int j = j1; j < j2; ++j)
      for (int i = i1; i < i2; ++i) {
        fieldSend[c] = phi[i][j][k][t];
        c += 1;
      }
  return;
}

void CopyRecvBuf(double ****phi, int t, int iStart, int iEnd, int jStart,
                 int jEnd, int kStart, int kEnd, int disp, int dir,
                 double *fieldRecv, int MaxBufLen) {
  int i1, i2, j1, j2, k1, k2, c;
  int i, j, k;

  if (dir < 0 || dir > 2) {
    std::cout << "CRB: dir is wrong\n";
    exit(1);
  }
  if (disp != 1 && disp != -1) {
    std::cout << "CRB: disp is wrong\n";
    exit(1);
  }

  if (dir == 0) {
    i1 = iStart + 1;
    i2 = iEnd - 1;
    j1 = jStart + 1;
    j2 = jEnd - 1;

    if (disp == 1)
      k1 = k2 = 0;
    else
      k1 = k2 = kEnd;
  } else if (dir == 1) {
    i1 = iStart + 1;
    i2 = iEnd - 1;
    k1 = kStart + 1;
    k2 = kEnd - 1;

    if (disp == 1)
      j1 = j2 = 0;
    else
      j1 = j2 = jEnd;
  } else if (dir == 2) {
    j1 = jStart + 1;
    j2 = jEnd - 1;
    k1 = kStart + 1;
    k2 = kEnd - 1;

    if (disp == 1)
      i1 = i2 = 0;
    else
      i1 = i2 = jEnd;
  }

  c = 1;
  for (int k = k1; k < k2; ++k)
    for (int j = j1; j < j2; ++j)
      for (int i = i1; i < i2; ++i) {
        phi[i][j][k][t] = fieldRecv[c];
        c += 1;
      }
  return;
}

void Jacobi_sweep(int nx, int ny, int nz, double ****phi, int t0, int t1,
                  int **udim, double h, double &maxdelta) {
  double rhs = 1.0;
  double one_over_six = 1. / 6.;

  int i, j, k;
  maxdelta = 0.;

  for (int k = udim[0][0]; k < udim[1][0]; ++k)
    for (int j = udim[0][1]; j < udim[1][1]; ++j)
      for (int i = udim[0][2]; i < udim[1][2]; ++i) {
        phi[i][j][k][t1] =
            (phi[i - 1][j][k][t0] + phi[i + 1][j][k][t0] +
             phi[i][j - 1][k][t0] + phi[i][j + 1][k][t0] +
             phi[i][j][k - 1][t0] + phi[i][j][k + 1][t0] + h * h * rhs) *
            one_over_six;
        maxdelta = std::max(maxdelta, abs(phi[i][j][k][t1] - phi[i][j][k][t0]));
      }
  return;
}

// from
// https://github.com/cpraveen/cfdlab/blob/ee0a423956f216d7120e338c4026391c48b219e5/vtk/vtk_struct.cc#L7
void write_rectilinear_grid(int id, int nx, int ny, int nz, double ****var,
                            double t, int c) {
  using namespace std;

  ofstream fout;
  char filename[64];
  sprintf(filename, "rect%d.vtk", id);
  fout.open(filename);
  fout << "# vtk DataFile Version 3.0" << endl;
  fout << "Cartesian grid" << endl;
  fout << "ASCII" << endl;
  fout << "DATASET RECTILINEAR_GRID" << endl;
  fout << "FIELD FieldData 2" << endl;
  fout << "TIME 1 1 double" << endl;
  fout << t << endl;
  fout << "CYCLE 1 1 int" << endl;
  fout << c << endl;
  fout << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
  fout << "X_COORDINATES " << nx << " float" << endl;
  for (int i = 0; i < nx; ++i)
    // TODO don't hardcode this!
    fout << i * 1. / (nx - 1) << " ";
  fout << endl;
  fout << "Y_COORDINATES " << ny << " float" << endl;
  for (int j = 0; j < ny; ++j)
    fout << j * 1. / (ny - 1) << " ";
  fout << endl;
  fout << "Z_COORDINATES " << nz << " float" << endl;
  for (int k = 0; k < nz; ++k)
    fout << k * 1. / (nz - 1) << " ";

  fout << "POINT_DATA " << nx * ny * nz << endl;
  fout << "SCALARS density float" << endl;
  fout << "LOOKUP_TABLE default" << endl;
  // no need for k-loop since nk=1
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i)
      for (int k = 0; k < nz; ++k)
        fout << var[i][j][k][(int)t] << " ";
    fout << endl;
  }
  fout.close();

  cout << "Wrote Cartesian grid into rect.vtk" << endl;
}
