#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include "algorithms.h"

struct HeapNode {
    int idx;      // index of the array
    int pos;      // position in the array
    tuwtype_t* ptr;    // pointer to the element
};

struct HeapCompare {
    bool operator()(const HeapNode& a, const HeapNode& b) const {
        // Compare the elements pointed by a.ptr and b.ptr
        return *(a.ptr) > *(b.ptr); // Min-heap
    }
};

int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);


    // 1. Allgather all data
    MPI_Allgather(
        sendbuf, sendcount, sendtype,
        recvbuf, recvcount, recvtype,
        comm
    );

    // 2. Prepare for p-way merge
    std::vector<HeapNode> heap;
    heap.reserve(size);

    for (int i = 0; i < size; ++i) {
        if (recvcount > 0) {
            heap.push_back(HeapNode{
                i, 0,
                static_cast<tuwtype_t*>(recvbuf) + i * recvcount
            });
        }
    }

    std::vector<int> positions(size, 0);
    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapCompare> pq(
        HeapCompare{}, heap
    );

    // Allocate temporary output buffer
    std::vector<tuwtype_t> output(size * recvcount);

    for (int out_idx = 0; out_idx < size * recvcount; ++out_idx) {
        if (pq.empty()) break;
        HeapNode min = pq.top(); pq.pop();

        // Copy the element to output
        std::memcpy(
            output.data() + out_idx,
            min.ptr,
            sizeof(tuwtype_t)
        );

        // Advance in the corresponding array
        positions[min.idx]++;
        if (positions[min.idx] < recvcount) {
            pq.push(HeapNode{
                min.idx,
                positions[min.idx],
                static_cast<tuwtype_t*>(recvbuf) + min.idx * recvcount + positions[min.idx]
            });
        }
    }

    // Copy merged output back to recvbuf
    std::memcpy(recvbuf, output.data(), size * recvcount * sizeof(tuwtype_t));

    return MPI_SUCCESS;
}
