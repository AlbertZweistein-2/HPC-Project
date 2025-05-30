#include <mpi.h>
#include <cmath>
#include <vector>
#include "algorithms.h"

void twowaymerge(void *X, void *Y, void *Z, int n) {
  tuwtype_t *x = static_cast<tuwtype_t *>(X);
  tuwtype_t *y = static_cast<tuwtype_t *>(Y);
  tuwtype_t *z = static_cast<tuwtype_t *>(Z);

  int i = 0, j = 0, k = 0;
  while (i < n && j < n) {
    if (x[i] <= y[j]) {
      z[k++] = x[i++];
    } else {
      z[k++] = y[j++];
    }
  }
  while (i < n) {
    z[k++] = x[i++];
  }
  while (j < n) {
    z[k++] = y[j++];
  }
}

int HPC_AllgatherMergeCirculant(const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm) {
  int rank, size; // r, p
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size == 1) {
    std::memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));
    return MPI_SUCCESS;
  }

  int q = static_cast<int>(std::ceil(std::log2(size)));
  std::vector<int> s_k(q + 1, 0);
  s_k[q] = size;
  for (int i = q - 1; i >= 0; --i) {
      s_k[i] = static_cast<int>(std::ceil(s_k[i + 1] / 2.0));
  }

  for (int k = 0; k < q; k++)
  {
    int epsilon = s_k[k+1] & 0x1;
    int t = (rank - s_k[k] + epsilon + size) % size;
    int f = (rank + s_k[k] - epsilon) % size;

    tuwtype_t *sendbuf_W = static_cast<tuwtype_t *>(const_cast<void *>(sendbuf));
    tuwtype_t *recvbuf_T = static_cast<tuwtype_t *>(const_cast<void *>(sendbuf));

    if (epsilon == 1) {
      MPI_Sendrecv(sendbuf_W, sendcount, sendtype, t, 0,
                   recvbuf_T, sendcount, sendtype, f, 0, 
                   comm, MPI_STATUS_IGNORE);
      tuwtype_t *temp = static_cast<tuwtype_t *>(recvbuf);
      twowaymerge(sendbuf_W, recvbuf_T, temp, sendcount);
    }
    else {
      MPI_Sendrecv(sendbuf_W, sendcount, sendtype, t, 0,
                   recvbuf_T, sendcount, sendtype, f, 0, 
                   comm, MPI_STATUS_IGNORE);
      std::memcpy(static_cast<tuwtype_t *>(recvbuf) + rank * sendcount,
                  recvbuf_T, sendcount * sizeof(tuwtype_t));
    }
  }
                                
  return MPI_SUCCESS;
}
