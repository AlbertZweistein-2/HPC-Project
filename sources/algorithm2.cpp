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
  
  std::vector<tuwtype_t> V(sendbuf_typed, sendbuf_typed + sendcount);
  std::vector<tuwtype_t> M;
  std::vector<tuwtype_t> T;
  std::vector<tuwtype_t> merge_buffer;
  
  const int max_buffer_size = size * sendcount;
  M.reserve(max_buffer_size);
  T.reserve(max_buffer_size);
  merge_buffer.reserve(max_buffer_size);
  
  M.resize(sendcount);
  std::copy(V.begin(), V.end(), M.begin());
  size_t current_size = sendcount;

  for (int k = 0; k < q; k++) {
    int epsilon = s_k[k + 1] & 0x1;
    int t = (rank - s_k[k] + epsilon + size) % size;
    int f = (rank + s_k[k] - epsilon) % size;
    
    if (epsilon == 1) {
      if (T.size() < current_size)
        T.resize(current_size);
        
      MPI_Sendrecv(M.data(), current_size, sendtype, t, 0,
                  T.data(), current_size, recvtype, f, 0, 
                  comm, MPI_STATUS_IGNORE);
      
      if (merge_buffer.size() < 2 * current_size)
        merge_buffer.resize(2 * current_size);
        
      std::merge(M.begin(), M.begin() + current_size, 
                T.begin(), T.begin() + current_size, 
                merge_buffer.begin());
      
      std::swap(M, merge_buffer);
      M.resize(2 * current_size);
      current_size *= 2;
      
    } else {
      if (k == 0) {
        T.resize(sendcount);
          
        MPI_Sendrecv(V.data(), sendcount, sendtype, t, 0,
                    T.data(), sendcount, recvtype, f, 0, 
                    comm, MPI_STATUS_IGNORE);

        std::swap(M, T);
        M.resize(sendcount);
        current_size = sendcount;
        
      } else {
        if (merge_buffer.size() < current_size + sendcount)
          merge_buffer.resize(current_size + sendcount);
        
        std::merge(V.begin(), V.end(), 
                  M.begin(), M.begin() + current_size, 
                  merge_buffer.begin());
        
        if (T.size() < current_size + sendcount)
          T.resize(current_size + sendcount);
        
        MPI_Sendrecv(merge_buffer.data(), current_size + sendcount, sendtype, t, 0,
                    T.data(), current_size + sendcount, recvtype, f, 0, 
                    comm, MPI_STATUS_IGNORE);
        
        if (merge_buffer.size() < current_size * 2 + sendcount)
          merge_buffer.resize(current_size * 2 + sendcount);
        
        std::merge(M.begin(), M.begin() + current_size,
                  T.begin(), T.begin() + current_size + sendcount,
                  merge_buffer.begin());
        
        std::swap(M, merge_buffer);
        M.resize(current_size * 2 + sendcount);
        current_size = current_size * 2 + sendcount;
      }
    }
  }

  std::merge(V.begin(), V.end(),
            M.begin(), M.begin() + current_size,
            static_cast<tuwtype_t*>(recvbuf));

  return MPI_SUCCESS;
}