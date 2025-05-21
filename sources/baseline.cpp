#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include "algorithms.h"

struct HeapNode {
    int idx;      // index of the array
    int pos;      // position in the array
    char* ptr;    // pointer to the element
};

struct HeapCompare {
    MPI_Datatype type;
    int typesize;
    bool operator()(const HeapNode& a, const HeapNode& b) const {
        // Compare the elements pointed by a.ptr and b.ptr
        return std::memcmp(a.ptr, b.ptr, typesize) > 0;
    }
};

int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    // 1. Allgather all data
    MPI_Allgather(
        sendbuf, sendcount, sendtype,
        recvbuf, recvcount, recvtype,
        comm
    );

    // 2. Prepare for p-way merge
    // Each process's data is at ((char*)recvbuf) + i * recvcount * typesize
    std::vector<HeapNode> heap;
    heap.reserve(size);

    for (int i = 0; i < size; ++i) {
        if (recvcount > 0) {
            heap.push_back(HeapNode{
                i, 0,
                ((char*)recvbuf) + i * recvcount * typesize
            });
        }
    }

    std::vector<int> positions(size, 0);
    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapCompare> pq(
        HeapCompare{recvtype, typesize}, heap
    );

    // Allocate temporary output buffer
    std::vector<char> output(size * recvcount * typesize);

    for (int out_idx = 0; out_idx < size * recvcount; ++out_idx) {
        if (pq.empty()) break;
        HeapNode min = pq.top(); pq.pop();

        // Copy the element to output
        std::memcpy(
            output.data() + out_idx * typesize,
            min.ptr,
            typesize
        );

        // Advance in the corresponding array
        positions[min.idx]++;
        if (positions[min.idx] < recvcount) {
            pq.push(HeapNode{
                min.idx,
                positions[min.idx],
                ((char*)recvbuf) + min.idx * recvcount * typesize + positions[min.idx] * typesize
            });
        }
    }

    // Copy merged output back to recvbuf
    std::memcpy(recvbuf, output.data(), size * recvcount * typesize);

    return MPI_SUCCESS;
}
