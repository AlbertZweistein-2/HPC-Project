// Version description:
// This is the third version of brucks allgather algorithm
// It sends the unsorted and the merged arrays to the next process and also receives the unsorted and merged arrays from the previous process.
// It then does 2-way merge on the received sorted arrays and the local sorted array,
// except for the last round, where the unsorted remaining arrays are sent and received,
// and then k-way merged, except the last send only includes one remaining array.

#include "mpi.h"
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
    // abort if sendcount is not equal to recvcount
    if (sendcount != recvcount) {
        std::cerr << "Error: sendcount must be equal to recvcount in HPC_AllgatherMergeBruck" << std::endl;
        return MPI_ERR_COUNT;
    }
    int r, p;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &p);

    // Bruck Gathering algorithm
    size_t q = static_cast<int>(ceil(log2(p)));
    size_t s_k, t, f;

    // Initialisiere recvbuf mit lokalen Daten
    memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));

    // M enthält die gemergten Daten, wächst im Verlauf!
    vector<tuwtype_t> M;
    M.reserve(p * sendcount); // Maximale mögliche Größe
    M.assign(static_cast<const tuwtype_t*>(sendbuf),
             static_cast<const tuwtype_t*>(sendbuf) + sendcount);

    // Hilfspuffer für Merge (um Allokationen zu sparen)
    vector<tuwtype_t> M_new;

    for (size_t k = 0; k < q; k++) {
        s_k = 1 << k;
        if (k == q-1)
            s_k -= ((1 << q) - p);
        t = (r - (1 << k) + p) % p; // process number to send to
        f = (r + (1 << k)) % p;     // process number to receive from

        // --- Sendpuffer aufbauen: gemerged + ungemerged (aus recvbuf!) ---
        vector<tuwtype_t> sendbuf_temp;
        sendbuf_temp.reserve(M.size() + s_k * sendcount);
        sendbuf_temp.assign(M.begin(), M.end());
        sendbuf_temp.insert(sendbuf_temp.end(),
            static_cast<tuwtype_t*>(recvbuf),
            static_cast<tuwtype_t*>(recvbuf) + s_k * sendcount);

        // --- Recv-Buffer vorbereiten ---
        vector<tuwtype_t> recvbuf_temp(M.size() + s_k * sendcount);

        // --- Kommunikation ---
        MPI_Sendrecv(
            sendbuf_temp.data(), sendbuf_temp.size(), sendtype, t, 0,
            recvbuf_temp.data(), recvbuf_temp.size(), recvtype, f, 0,
            comm, MPI_STATUS_IGNORE
        );

        const tuwtype_t* remote_M       = recvbuf_temp.data();
        const tuwtype_t* remote_recvbuf = recvbuf_temp.data() + M.size();

        // --- Empfangene ungemergte Blöcke im lokalen recvbuf sichern ---
        tuwtype_t* local_slot = static_cast<tuwtype_t*>(recvbuf) + ((1 << k) * sendcount);
        std::memcpy(local_slot, remote_recvbuf, s_k * sendcount * sizeof(tuwtype_t));

        // --- Merging ---
        if (k < q - 1) {
            // 2-way merge mit remote_M
            M_new.resize(2 * M.size());
            std::merge(
                M.begin(), M.end(),
                remote_M, remote_M + M.size(),
                M_new.begin()
            );
            M.swap(M_new);
        } else if (s_k == 1) {
            // 2-way merge mit remote_recvbuf
            M_new.resize(M.size() + s_k * sendcount);
            std::merge(
                M.begin(), M.end(),
                remote_recvbuf, remote_recvbuf + s_k * sendcount,
                M_new.begin()
            );
            M.swap(M_new);
        } else {
            // K-way merge
            vector<tuwtype_t*> sources(s_k + 1);
            vector<size_t> lengths(s_k + 1);
            sources[0] = M.data();
            lengths[0] = M.size();
            for (size_t i = 0; i < s_k; ++i) {
                sources[i + 1] = static_cast<tuwtype_t*>(recvbuf) + ((1 << k) + i) * sendcount;
                lengths[i + 1] = sendcount;
            }
            M_new.resize(M.size() + s_k * sendcount);
            kway_merge_variable_lengths(sources.data(), lengths.data(), s_k + 1, M_new.data());
            M.swap(M_new);
        }
    }

    // Copy back merged data into recvbuf
    std::copy(M.begin(), M.end(), static_cast<tuwtype_t*>(recvbuf));
    return MPI_SUCCESS;
}
