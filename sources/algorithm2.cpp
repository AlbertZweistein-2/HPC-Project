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
  int rank, size; 
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
  
  const tuwtype_t* sendbuf_typed = static_cast<const tuwtype_t*>(sendbuf);
  
  // Initialize buffers with proper sizes
  std::vector<tuwtype_t> V(sendbuf_typed, sendbuf_typed + sendcount);
  std::vector<tuwtype_t> M(sendcount);
  std::vector<tuwtype_t> T(sendcount);
  std::vector<tuwtype_t> W_prime(2 * sendcount);
  
  std::copy(V.begin(), V.end(), M.begin());
  int current_size = sendcount;

  for (int k = 0; k < q; k++) {
    int epsilon = s_k[k + 1] & 0x1;
    int t = (rank - s_k[k] + epsilon + size) % size;
    int f = (rank + s_k[k] - epsilon) % size;
    
    T.resize(current_size);
    
    if (epsilon == 1) {

      MPI_Sendrecv(M.data(), current_size, sendtype, t, 0,
                  T.data(), current_size, recvtype, f, 0, 
                  comm, MPI_STATUS_IGNORE);
      
      std::vector<tuwtype_t> temp(current_size * 2);
      
      std::merge(M.begin(), M.begin() + current_size, 
                T.begin(), T.begin() + current_size, 
                temp.begin());
      
      M = temp;
      current_size *= 2;
      
    } else {
      if (k == 0) {

        MPI_Sendrecv(V.data(), sendcount, sendtype, t, 0,
                    T.data(), sendcount, recvtype, f, 0, 
                    comm, MPI_STATUS_IGNORE);

        M = T;
        current_size = sendcount;
        
      } else {

        W_prime.resize(current_size + sendcount);

        std::merge(V.begin(), V.end(), 
                  M.begin(), M.begin() + current_size, 
                  W_prime.begin());
        
        T.resize(current_size + sendcount);
        
        MPI_Sendrecv(W_prime.data(), current_size + sendcount, sendtype, t, 0,
                    T.data(), current_size + sendcount, recvtype, f, 0, 
                    comm, MPI_STATUS_IGNORE);

        std::vector<tuwtype_t> temp(2 * current_size + sendcount);
        
        std::merge(M.begin(), M.begin() + current_size,
                  T.begin(), T.begin() + current_size + sendcount,
                  temp.begin());
        
        M = temp;
        current_size = current_size * 2 + sendcount;
      }
    }
  }

  std::merge(V.begin(), V.end(),
            M.begin(), M.begin() + current_size,
            static_cast<tuwtype_t*>(recvbuf));

  return MPI_SUCCESS;
}