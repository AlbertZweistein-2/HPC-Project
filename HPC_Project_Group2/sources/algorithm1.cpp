// Version description:
// This is the fourth version of brucks allgather algorithm
// Only uses 2-way merges, no k-way merges.
// Uses a two-way merge with tracking of process origins.

#include "mpi.h"
#include "algorithms.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

using namespace std;

struct TrackedElement {
    tuwtype_t value;
    int origin;

    bool operator<(const TrackedElement& other) const {
        return value < other.value;
    }
};

void merge_with_tracking(const TrackedElement* list1, int len1,
                         const TrackedElement* list2, int len2,
                         TrackedElement* result){
    int i = 0, j = 0, k = 0;

    //Standard 2-way merge, but with tracking
    while (i < len1 && j < len2) {
        if(list1[i] < list2[j]) {
            result[k++] = list1[i++];
        }
        else {
            result[k++] = list2[j++];
        }
    }

    while (i < len1) {
        result[k++] = list1[i++];
    }
    while (j < len2) {
        result[k++] = list2[j++];
    }

}

int HPC_AllgatherMergeBruck(const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                            void * recvbuf, int recvcount, MPI_Datatype recvtype,
                            MPI_Comm comm) {
    if (sendcount != recvcount) {
        std::cerr << "Error: sendcount must be equal to recvcount" << std::endl;
        return MPI_ERR_COUNT;
    }
    int r, p;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &r);

    int q = static_cast<int>(ceil(log2(p)));

    vector<TrackedElement> buffer_B(p * sendcount);
    
    vector<TrackedElement> recvbuf_tracked(p * sendcount);

    vector<TrackedElement> sendbuffer(p * sendcount);
    vector<TrackedElement> recvbuffer(p * sendcount);

    const tuwtype_t* local_data = static_cast<const tuwtype_t*>(sendbuf);
    for (int i = 0; i < sendcount; ++i) {
        recvbuf_tracked[i].value = local_data[i];
        recvbuf_tracked[i].origin = r;
    }

    TrackedElement* curr = recvbuf_tracked.data();
    TrackedElement* next = buffer_B.data();
    int curr_len = sendcount;
    int s_k, t, f;
    
    // Neue Hilfsvariablen
    TrackedElement* sendptr;
    int send_len;

    for (int k = 0; k < q; ++k) {
        s_k = 1 << k;
        t = (r - s_k + p) % p;
        f = (r + s_k) % p; 

        if (k == q-1) {
            s_k -= ((1 << q) - p);
            
            send_len = 0;
            for (int i = 0; i < curr_len; ++i) {
                if (int(curr[i].origin - r + p) % p < s_k) {
                    sendbuffer[send_len++] = curr[i];
                }
            }
            sendptr = sendbuffer.data();
        } else {
            sendptr = curr;
            send_len = curr_len;
        }

        MPI_Sendrecv(
            sendptr, send_len * sizeof(TrackedElement), MPI_BYTE, t, 0,
            recvbuffer.data(), send_len * sizeof(TrackedElement), MPI_BYTE, f, 0,
            comm, MPI_STATUS_IGNORE
        );

        merge_with_tracking(curr, curr_len, recvbuffer.data(), send_len, next);
        curr_len += send_len;

        std::swap(curr, next);
    }

    tuwtype_t* result = static_cast<tuwtype_t*>(recvbuf);
    for (int i = 0; i < curr_len; ++i) {
        result[i] = curr[i].value;
    }
    return MPI_SUCCESS;
}