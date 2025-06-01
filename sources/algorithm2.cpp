#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include "algorithms.h"

int HPC_AllgatherMergeCirculant(const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm) {
  int rank, size; // r, p
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Handle single process case
  if (size == 1) {
    memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));
    return MPI_SUCCESS;
  }

  // Calculate q and s_k values
  int q = static_cast<int>(std::ceil(std::log2(size)));
  std::vector<int> s_k(q + 1, 0);
  s_k[q] = size;
  for (int i = q - 1; i >= 0; --i) {
      s_k[i] = static_cast<int>(std::ceil(s_k[i + 1] / 2.0));
  }

  // Cast sendbuf to proper type
  const tuwtype_t* sendbuf_typed = static_cast<const tuwtype_t*>(sendbuf);
  
  // Initialize buffers with proper sizes
  std::vector<tuwtype_t> V(sendbuf_typed, sendbuf_typed + sendcount);
  std::vector<tuwtype_t> W(sendcount);
  std::vector<tuwtype_t> T(sendcount);
  std::vector<tuwtype_t> W_prime(2 * sendcount);
  
  // Initial copy of own data
  std::copy(V.begin(), V.end(), W.begin());
  int current_size = sendcount;

  for (int k = 0; k < q; k++) {
    int epsilon = s_k[k + 1] & 0x1;
    int t = (rank - s_k[k] + epsilon + size) % size;
    int f = (rank + s_k[k] - epsilon) % size;
    
    // Resize T to receive the appropriate amount of data
    T.resize(current_size);
    
    if (epsilon == 1) { // sk+1 is odd
      // Send W to process t and receive T from process f
      MPI_Sendrecv(W.data(), current_size, sendtype, t, 0,
                  T.data(), current_size, recvtype, f, 0, 
                  comm, MPI_STATUS_IGNORE);
      
      // Prepare result buffer
      std::vector<tuwtype_t> result(current_size * 2);
      
      // Merge W and T
      std::merge(W.begin(), W.begin() + current_size, 
                T.begin(), T.begin() + current_size, 
                result.begin());
      
      // Update W and its size
      W = result;
      current_size *= 2;
      
    } else { // sk+1 is even
      if (k == 0) {
        // First round: Send V to process t and receive W from process f
        MPI_Sendrecv(V.data(), sendcount, sendtype, t, 0,
                    T.data(), sendcount, recvtype, f, 0, 
                    comm, MPI_STATUS_IGNORE);
        
        // Copy T to W (first received data)
        W = T;
        current_size = sendcount;
        
      } else {
        // Subsequent rounds
        // Merge V and W into W_prime
        W_prime.resize(current_size + sendcount);
        std::merge(V.begin(), V.end(), 
                  W.begin(), W.begin() + current_size, 
                  W_prime.begin());
        
        // Resize T to receive the merged data
        T.resize(current_size + sendcount);
        
        // Send W_prime to process t and receive T from process f
        MPI_Sendrecv(W_prime.data(), current_size + sendcount, sendtype, t, 0,
                    T.data(), current_size + sendcount, recvtype, f, 0, 
                    comm, MPI_STATUS_IGNORE);
        
        // Prepare result buffer
        std::vector<tuwtype_t> result(2 * current_size + sendcount);
        
        // Merge W and T
        std::merge(W.begin(), W.begin() + current_size,
                  T.begin(), T.begin() + current_size + sendcount,
                  result.begin());
        
        // Update W and its size
        W = result;
        current_size = current_size * 2 + sendcount;
      }
    }
  }

  std::merge(V.begin(), V.end(),
            W.begin(), W.begin() + current_size,
            static_cast<tuwtype_t*>(recvbuf));

  
  return MPI_SUCCESS;
}