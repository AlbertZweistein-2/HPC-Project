#include <mpi.h>
#include <cmath>
#include <vector>
#include "algorithms.h"

#include <iostream>

void twowaymerge(const void *X, const void *Y, void *Z, int n) {
  const tuwtype_t *x = static_cast<const tuwtype_t *>(X);
  const tuwtype_t *y = static_cast<const tuwtype_t *>(Y);
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
    memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));
    return MPI_SUCCESS;
  }

  int q = static_cast<int>(std::ceil(std::log2(size)));
  std::vector<int> s_k(q + 1, 0);
  s_k[q] = size;
  for (int i = q - 1; i >= 0; --i) {
      s_k[i] = static_cast<int>(std::ceil(s_k[i + 1] / 2.0));
  }

  // Temporary buffers for merging
  std::vector<tuwtype_t> W(recvcount, 0);
  std::vector<tuwtype_t> T(recvcount, 0);
  std::vector<tuwtype_t> W_prime(recvcount, 0);

  // Copy initial data into W
  memmove(W.data(), sendbuf, sendcount * sizeof(tuwtype_t));

  for (int k = 0; k < q; k++) {
    int epsilon = s_k[k + 1] & 0x1;
    int t = (rank - s_k[k] + epsilon + size) % size;
    int f = (rank + s_k[k] - epsilon) % size;

    if (epsilon == 1) {
      // Send W to process t and receive T from process f
      MPI_Sendrecv(W.data(), sendcount, sendtype, t, 0,
                   T.data(), sendcount, sendtype, f, 0, 
                   comm, MPI_STATUS_IGNORE);

      // Merge W and T into W
      twowaymerge(W.data(), T.data(), W.data(), recvcount);
    } else {
      if (k == 0) {
        // Send V (initial data) to process t and receive W from process f
        MPI_Sendrecv(sendbuf, sendcount, sendtype, t, 0,
                     W.data(), sendcount, sendtype, f, 0, 
                     comm, MPI_STATUS_IGNORE);
      } else {
        // Merge V and W into W_prime
        twowaymerge(sendbuf, W.data(), W_prime.data(), recvcount);

        // Send W_prime to process t and receive T from process f
        MPI_Sendrecv(W_prime.data(), recvcount, sendtype, t, 0,
                     T.data(), recvcount, sendtype, f, 0, 
                     comm, MPI_STATUS_IGNORE);

        // Merge W and T into W
        twowaymerge(W.data(), T.data(), W.data(), recvcount);
      }
    }
  }

  // Final merge of V and W into W
  twowaymerge(sendbuf, W.data(), W_prime.data(), recvcount);

  // Copy the final result into recvbuf
  memcpy(recvbuf, W_prime.data(), recvcount * sizeof(tuwtype_t));

  return MPI_SUCCESS;
}
