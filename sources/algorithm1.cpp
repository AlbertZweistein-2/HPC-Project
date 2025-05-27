#include <mpi.h>

#include "algorithms.h"

#include <cmath>

using namespace std;

int HPC_AllgatherMergeBruck(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf, int recvcount,
                            MPI_Datatype recvtype, MPI_Comm comm) {
  
  int r, p;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &p);

  //Bruck Gathering algorithm
  int q = static_cast<int>(ceil(log2(p)));
  vector<tuwtype_t> M(sendcount);
  memcpy(M.data(), sendbuf, sendcount * sizeof(tuwtype_t));
  
  for (int k = 0; k < q; k++) {
    int s_k = 1 << k; // 2^k
    int recvsize = (1 << k) * sendcount; // 2^k * sendcount
    size_t t = (r - s_k + p) % p; // process number to send to
    size_t f = (r + s_k ) % p; // process number to receive from

    vector<tuwtype_t> recvbuf_temp(recvsize);

    MPI_Sendrecv(M.data(), recvsize, sendtype, t, 0,
                 recvbuf_temp.data(), recvsize, recvtype, f, 0,
                 comm, MPI_STATUS_IGNORE);
    // Merge the received data into M using std::merge (both vectors must be sorted)
    merge(M.begin(), M.end(), recvbuf_temp.begin(), recvbuf_temp.end(), static_cast<tuwtype_t*>(recvbuf));
    M.assign(static_cast<tuwtype_t*>(recvbuf), 
             static_cast<tuwtype_t*>(recvbuf) + M.size() + recvsize);


  }
  memcpy(recvbuf, M.data(), M.size() * sizeof(tuwtype_t));
  return MPI_SUCCESS;
}
