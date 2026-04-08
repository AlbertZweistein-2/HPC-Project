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

// Merge für Listen mit unterschiedlichen Längen
void kway_merge_variable_lengths(tuwtype_t** sources, const size_t* lengths, int num_lists, tuwtype_t* dest) {
    struct HeapNode {
        int idx;      // Index der Liste
        int pos;      // Position in der Liste
        tuwtype_t* ptr; // Zeiger auf das aktuelle Element
    };
    struct HeapCompare {
        bool operator()(const HeapNode& a, const HeapNode& b) const {
            return *(a.ptr) > *(b.ptr);  // Min-Heap
        }
    };

    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapCompare> heap{HeapCompare{}};
    for (int i = 0; i < num_lists; ++i) {
        if (lengths[i] > 0) {
            heap.push(HeapNode{i, 0, sources[i]});
        }
    }
    int merged_offset = 0;
    while (!heap.empty()) {
        HeapNode node = heap.top();
        heap.pop();

        dest[merged_offset++] = *node.ptr;

        if (node.pos + 1 < (int)lengths[node.idx]) {
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
    int s_k;
    memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));
    // Initialisiere zwei Puffer: recvbuf und tempbuf
    tuwtype_t* buffer1 = static_cast<tuwtype_t*>(recvbuf);
    std::vector<tuwtype_t> tempbuf(p * recvcount);
    tuwtype_t* buffer2 = tempbuf.data();
    tuwtype_t* M = buffer1;
    tuwtype_t* M_new = buffer2;
    size_t M_len = sendcount;
    memcpy(M, sendbuf, sendcount * sizeof(tuwtype_t));
    size_t send_recv_size;
    size_t t, f;

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
        if(s_k == 1){
            std::merge(
                M, M + ((1 << k) * sendcount),
                static_cast<tuwtype_t*>(recvbuf) + ((1 << k) * sendcount),
                static_cast<tuwtype_t*>(recvbuf) + ((1 << k) + 1) * sendcount,
                M_new
            );
            M_len += sendcount;
        }else{
            std::vector<tuwtype_t*> sources(s_k + 1);
            std::vector<size_t> lengths(s_k + 1);
            sources[0] = M;
            lengths[0] = (1 << k) * sendcount;
            for (int i = 0; i < s_k; ++i) {
                sources[i + 1] = static_cast<tuwtype_t*>(recvbuf) + ((1 << k) + i) * sendcount;
                lengths[i + 1] = sendcount;
            }
            kway_merge_variable_lengths(sources.data(), lengths.data(), s_k + 1, M_new);
            M_len += s_k * sendcount;
        }
        // Pointer tauschen, kein Kopieren!
        std::swap(M, M_new);
    }
    // Am Ende: falls M nicht auf recvbuf zeigt, kopiere zurück
    if (M != static_cast<tuwtype_t*>(recvbuf)) {
        memcpy(recvbuf, M, M_len * sizeof(tuwtype_t));
    }
    return MPI_SUCCESS;
}
