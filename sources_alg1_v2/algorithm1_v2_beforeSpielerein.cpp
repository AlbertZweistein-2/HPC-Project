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
    vector<tuwtype_t> M(p * recvcount);
    copy(static_cast<const tuwtype_t*>(sendbuf), static_cast<const tuwtype_t*>(sendbuf) + sendcount, M.data());
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
        if(s_k == 1){
            // Schneller 2-Wege-Merge
            std::vector<tuwtype_t> M_new(((1 << k) + 1) * sendcount);
            std::merge(
                M.begin(), M.begin() + ((1 << k) * sendcount),
                static_cast<tuwtype_t*>(recvbuf) + ((1 << k) * sendcount),
                static_cast<tuwtype_t*>(recvbuf) + ((1 << k) + 1) * sendcount,
                M_new.begin()
            );
            M = std::move(M_new);
        }else{
            // K-way merge wie gehabt
            vector<tuwtype_t*> sources(s_k + 1);
            vector<size_t> lengths(s_k + 1);

            sources[0] = M.data();
            lengths[0] = (1 << k) * sendcount;
            for (int i = 0; i < s_k; ++i) {
                sources[i + 1] = static_cast<tuwtype_t*>(recvbuf) + ((1 << k) + i) * sendcount;
                lengths[i + 1] = sendcount;
            }
            vector<tuwtype_t> M_new(((1 << k) + s_k) * sendcount);
            kway_merge_variable_lengths(sources.data(), lengths.data(), s_k + 1, M_new.data());
            M = std::move(M_new);
        }
    }
    // Copy back merged data into recvbuf
    copy(M.begin(), M.end(), static_cast<tuwtype_t*>(recvbuf));
    return MPI_SUCCESS;
}
