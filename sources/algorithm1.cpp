#include <mpi.h>

#include "algorithms.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

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
    size_t t = (r - s_k + p) % p; // process number to send to
    size_t f = (r + s_k ) % p; // process number to receive from
    int send_recv_size, offset = 0;
    vector<tuwtype_t> recvbuf_temp;
      if (k == q-1) {
        int overflow = ( (1<<q) - p );
        int valid_blocks = s_k - overflow;
        send_recv_size = valid_blocks * sendcount;
        offset = overflow * sendcount;
        //print the sent data for process 2
        if (r == 2) {
          //print M before sending
          cout << "Process " << r << " M before sending: ";
          for (const auto& val : M) cout << val << " ";
          cout << endl;
          cout << "Process " << r << " sending to " << t << " with offset: " << offset << endl;
          for (int i = offset; i < offset + send_recv_size; ++i) {
            cout << M[i] << " ";
          }
          cout << endl;
        }
      } else {
        send_recv_size = s_k * sendcount;
      }

    recvbuf_temp.resize(send_recv_size);
    MPI_Sendrecv(
      M.data() + offset, send_recv_size, sendtype, t, 0,
      recvbuf_temp.data(), send_recv_size, recvtype, f, 0,
      comm, MPI_STATUS_IGNORE
    );
    
    // Merge the received data into M using std::merge (both vectors must be sorted)
    merge(M.begin(), M.end(), recvbuf_temp.begin(), recvbuf_temp.end(), static_cast<tuwtype_t*>(recvbuf));
    M.assign(static_cast<tuwtype_t*>(recvbuf), 
             static_cast<tuwtype_t*>(recvbuf) + M.size() + send_recv_size);
    if(r == 3){
      cout << "Process " << r << " received from " << f << ": ";
      for (const auto& val : recvbuf_temp) std::cout << val << " ";
      cout << endl;
      cout << "-------"<< endl;
      cout << "Process " << r << " after merge: ";
      for (const auto& val : M) std::cout << val << " ";
      cout << endl;
      cout << "-------"<< std::endl;
    }
  }
  // print M
  // std::cout << "Process " << r << " M: ";
  // for (const auto& val : M) std::cout << val << " ";
  //   std::cout << std::endl;
  memcpy(recvbuf, M.data(), p * recvcount * sizeof(tuwtype_t));
  return MPI_SUCCESS;
}
