#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
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

  /* Debug print for s_k
  for (int i = 0; i < q; ++i) {
    std::cout << "s_k[" << i << "] = " << s_k[i] << std::endl;
  }
  */

  // Temporary buffers for merging
  std::vector<tuwtype_t> T;
  T.reserve(size * sendcount);
  std::vector<tuwtype_t> W;
  W.reserve(size * sendcount);
  std::vector<tuwtype_t> W_prime;
  W_prime.reserve(size * sendcount);

  for (int k = 0; k < q; k++) {
    int epsilon = s_k[k + 1] & 0x1;
    int t = (rank - s_k[k] + epsilon + size) % size;
    int f = (rank + s_k[k] - epsilon) % size;

    // Debug print for current step
    /*
    std::cout << "Rank " << rank << ", k = " << k 
              << ", t = " << t << ", f = " << f 
              << ", epsilon = " << epsilon << std::endl;
    */

    if (epsilon == 1) {
      // Send W to process t and receive T from process f
      MPI_Sendrecv(W.data(), sendcount, sendtype, t, 0,
                   T.data(), sendcount, sendtype, f, 0, 
                   comm, MPI_STATUS_IGNORE);

      // Merge W and T into M
      std::merge(W.begin(), W.end(), T.begin(), T.end(), W.begin());

    } else {
      if (k == 0) {
        // Send V (initial data) to process t and receive W from process f
        MPI_Sendrecv(static_cast<const tuwtype_t*>(sendbuf), sendcount, sendtype, t, 0,
                     W.data(), sendcount, sendtype, f, 0, 
                     comm, MPI_STATUS_IGNORE);

        // Debug
        if (rank == 0) {
          std::cout << "Rank " << rank << ", initial W size: " 
                    << W.size() << ", contents: ";
          for (const auto& item : W) {
            std::cout << item << " ";
          }
          std::cout << std::endl;
        }
      } else {
        // Merge V and W into W_prime
        std::merge(
          static_cast<const tuwtype_t*>(sendbuf),
          static_cast<const tuwtype_t*>(sendbuf) + sendcount,
          W.begin(),
          W.end(),
          W_prime.begin()
        );

        // Send W_prime to process t and receive T from process f
        MPI_Sendrecv(W_prime.data(), W_prime.size(), sendtype, t, 0,
                     T.data(), W_prime.size(), sendtype, f, 0, 
                     comm, MPI_STATUS_IGNORE);

        // Merge W and T into W
        std::merge(W.begin(), W.end(), T.begin(), T.end(), W.data());
      }
    }

    // Debug print for merged W
    if (rank == 0) {
      std::cout << "Rank " << rank << ", after merge W size: " 
                << W.size() << ", contents: ";
      for (const auto& item : W) {
        std::cout << item << " ";
      }
      std::cout << std::endl;
    }
  }

  std::merge(
    static_cast<const tuwtype_t*>(sendbuf),
    static_cast<const tuwtype_t*>(sendbuf) + sendcount,
    W.begin(),
    W.end(),
    W.begin()
  );

  // Copy the final result into recvbuf
  memcpy(recvbuf, W.data(), size * sendcount * sizeof(tuwtype_t));

  return MPI_SUCCESS;
}
