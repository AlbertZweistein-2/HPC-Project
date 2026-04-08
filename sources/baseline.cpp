#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include "algorithms.h"
#include <iostream>


using namespace std;
struct HeapNode {
    int idx;      // index of the process array
    int pos;      // position in the array
    tuwtype_t* ptr;    // pointer to the element
};

struct HeapCompare {
    bool operator()(const HeapNode& a, const HeapNode& b) const {
        return *(a.ptr) > *(b.ptr);  // for a Min-Heap
    }
};

int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                           MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Allgather all data
    MPI_Allgather(
        sendbuf, sendcount, sendtype,
        recvbuf, recvcount, recvtype,
        comm
    );
    priority_queue<HeapNode, vector<HeapNode>, HeapCompare> heap{HeapCompare{}};
    vector<tuwtype_t*> sources(size); // Pointers to the start of each process's data

    for (int i = 0; i < size; ++i) {
        tuwtype_t* base = static_cast<tuwtype_t*>(recvbuf) + i * recvcount;
        sources[i] = base;
        if (recvcount > 0) {
            heap.push(HeapNode{i, 0, base});
        }
    }
    // 3. Merge result into temporary buffer
    vector<tuwtype_t> merged(size * recvcount);
    tuwtype_t* dest = merged.data();
    int merged_offset = 0;

    while (!heap.empty()) {
        HeapNode node = heap.top();
        heap.pop();

        dest[merged_offset] = *node.ptr;
        ++merged_offset;

        // Move pointer to next element in same list
        if (node.pos + 1 < recvcount) {
            node.pos += 1;
            node.ptr = sources[node.idx] + node.pos;
            heap.push(node);
        }
    }
    // 4. Copy back merged data into recvbuf
    copy(merged.begin(), merged.end(), static_cast<tuwtype_t*>(recvbuf));
    // memcpy(recvbuf, merged.data(), size * recvcount);

    return MPI_SUCCESS;
}
