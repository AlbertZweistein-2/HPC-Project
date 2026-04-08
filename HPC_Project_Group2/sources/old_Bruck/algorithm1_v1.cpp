// Version description:
// This implements the first version of the Algorithm 1
// Brucks Allgather Algorithm with a k-way merge at the end.
// Using Sendrecv for unordered elements, stored in R, and k-way merge at the end.

#include <mpi.h>
#include "algorithms.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <queue>

using namespace std;

struct HeapNode {
    int idx;
    int pos;
    tuwtype_t* ptr;
};

struct HeapCompare {
    bool operator()(const HeapNode& a, const HeapNode& b) const {
        return *(a.ptr) > *(b.ptr);  // for a Min-Heap
    }
};

void kway_merge(tuwtype_t** sources, int num_lists, int list_size, tuwtype_t* dest) {
    struct HeapNode {
        int idx;
        int pos;
        tuwtype_t* ptr;
    };
    struct HeapCompare {
        bool operator()(const HeapNode& a, const HeapNode& b) const {
            return *(a.ptr) > *(b.ptr);  // for a Min-Heap
        }
    };

    priority_queue<HeapNode, vector<HeapNode>, HeapCompare> heap{HeapCompare{}};
    for (int i = 0; i < num_lists; ++i) {
        if (list_size > 0) {
            heap.push(HeapNode{i, 0, sources[i]});
        }
    }
    int merged_offset = 0;
    while (!heap.empty()) {
        HeapNode node = heap.top();
        heap.pop();

        dest[merged_offset] = *node.ptr;
        ++merged_offset;

        if (node.pos + 1 < list_size) {
            node.pos += 1;
            node.ptr = sources[node.idx] + node.pos;
            heap.push(node);
        }
    }
}
int HPC_AllgatherMergeBruck(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf, int recvcount,
                            MPI_Datatype recvtype, MPI_Comm comm) {
    int r, p;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &p);

    // Bruck Gathering algorithm
    int q = static_cast<int>(ceil(log2(p)));
    memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));
    int s_k;
    size_t t, f, send_recv_size;
    for(int k=0; k < q; k++){
        s_k = 1 << k; // 2^k
        t = (r - s_k + p) % p; // process number to send to
        f = (r + s_k) % p; // process number to receive from
        if(k == q-1)
            s_k = s_k - ((1 << q) - p); // Adjust s_k for the last round
        send_recv_size = s_k * sendcount;
        MPI_Sendrecv(
            recvbuf, send_recv_size, sendtype, t, 0,
            static_cast<tuwtype_t*>(recvbuf) + ((1<<k)) * sendcount, send_recv_size, recvtype, f, 0,
            comm, MPI_STATUS_IGNORE
        );
    }

    // K-way merge the final result
    vector<tuwtype_t*> sources(p);
    for (int i = 0; i < p; ++i) {
        sources[i] = static_cast<tuwtype_t*>(recvbuf) + i * recvcount;
    }
    vector<tuwtype_t> merged(p * recvcount);
    kway_merge(sources.data(), p, recvcount, merged.data());

    // Copy back merged data into recvbuf
    copy(merged.begin(), merged.end(), static_cast<tuwtype_t*>(recvbuf));
    return MPI_SUCCESS;
}
