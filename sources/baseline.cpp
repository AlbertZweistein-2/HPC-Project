#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include "algorithms.h"

struct HeapNode {
    int idx;
    int pos;
    tuwtype_t* ptr;

    bool operator>(const HeapNode& other) const {
        return *(this->ptr) > *(other.ptr);
    }
};

int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                           MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Allgather(
        sendbuf, sendcount, sendtype,
        recvbuf, recvcount, recvtype,
        comm
    );

    std::vector<HeapNode> heap;
    heap.reserve(size);

    for (int i = 0; i < size; ++i) {
        heap.push_back(HeapNode{
            i, 0,
            static_cast<tuwtype_t*>(recvbuf) + i * recvcount
        });
    }

    std::vector<int> positions(size, 0);
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> pq(
        std::greater<HeapNode>{}, heap
    );

    std::vector<tuwtype_t> output(size * recvcount);

    for (int out_idx = 0; out_idx < size * recvcount; ++out_idx) {
        if (pq.empty()) break;
        HeapNode min = pq.top(); pq.pop();

        std::memcpy(
            output.data() + out_idx,
            min.ptr,
            sizeof(tuwtype_t)
        );

        positions[min.idx]++;
        if (positions[min.idx] < recvcount) {
            pq.push(HeapNode{
                min.idx,
                positions[min.idx],
                static_cast<tuwtype_t*>(recvbuf) + min.idx * recvcount + positions[min.idx]
            });
        }
    }

    std::memcpy(recvbuf, output.data(), size * recvcount * sizeof(tuwtype_t));

    return MPI_SUCCESS;
}
