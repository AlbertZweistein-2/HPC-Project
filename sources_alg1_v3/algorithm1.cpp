// Version description:
// This is the third version of brucks allgather algorithm
// Optimiert: verwendet zwei feste Puffer für Merge, arbeitet nur mit Pointer-Umschaltung.
// Minimale Heap-Fragmentierung, schnelle Merge-Schritte.

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
        int idx;
        int pos;
        tuwtype_t* ptr;
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

    if (sendcount != recvcount) {
        std::cerr << "Error: sendcount must be equal to recvcount in HPC_AllgatherMergeBruck" << std::endl;
        return MPI_ERR_COUNT;
    }
    int r, p;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &p);

    size_t q = static_cast<size_t>(ceil(log2(p)));
    size_t s_k, t, f;

    // Zwei große Puffer, die abwechselnd benutzt werden:
    vector<tuwtype_t> M_A(p * sendcount);
    vector<tuwtype_t> M_B(p * sendcount);
    tuwtype_t* curr = M_A.data();
    tuwtype_t* next = M_B.data();
    size_t curr_len = sendcount;

    // Initialisiere curr mit lokalen Daten
    memcpy(curr, sendbuf, sendcount * sizeof(tuwtype_t));
    memcpy(recvbuf, sendbuf, sendcount * sizeof(tuwtype_t));

    // Einmalige (maximal große) Puffer für Send/Recv, werden in der Schleife zurechtgestutzt:
    vector<tuwtype_t> sendbuf_temp(p * sendcount);
    vector<tuwtype_t> recvbuf_temp(p * sendcount);

    for (size_t k = 0; k < q; k++) {
        s_k = 1 << k;
        if (k == q-1)
            s_k -= ((1 << q) - p);
        t = (r - (1 << k) + p) % p;
        f = (r + (1 << k)) % p;

        size_t merged_len = curr_len;
        size_t unmerged_len = s_k * sendcount;
        size_t total_len = merged_len + unmerged_len;

        // Sende: [ curr | lokale unsortierte Blöcke ]
        std::copy(curr, curr + merged_len, sendbuf_temp.begin());
        std::copy(
            static_cast<tuwtype_t*>(recvbuf),
            static_cast<tuwtype_t*>(recvbuf) + unmerged_len,
            sendbuf_temp.begin() + merged_len
        );

        // Recv temporär anlegen (benutze immer nur total_len)
        MPI_Sendrecv(
            sendbuf_temp.data(), total_len, sendtype, t, 0,
            recvbuf_temp.data(), total_len, recvtype, f, 0,
            comm, MPI_STATUS_IGNORE
        );

        const tuwtype_t* remote_M = recvbuf_temp.data();
        const tuwtype_t* remote_recvbuf = recvbuf_temp.data() + merged_len;
        tuwtype_t* local_slot = static_cast<tuwtype_t*>(recvbuf) + ((1 << k) * sendcount);
        std::memcpy(local_slot, remote_recvbuf, unmerged_len * sizeof(tuwtype_t));

        // --- Merge-Schritt ---
        if (k < q - 1) {
            // 2-way merge mit remote_M
            std::merge(
                curr, curr + merged_len,
                remote_M, remote_M + merged_len,
                next
            );
            curr_len = 2 * merged_len;
        } else if (s_k == 1) {
            // 2-way merge mit remote_recvbuf
            std::merge(
                curr, curr + merged_len,
                remote_recvbuf, remote_recvbuf + unmerged_len,
                next
            );
            curr_len = merged_len + unmerged_len;
        } else {
            cout << "K-way merge in Process " << r << " at step " << k << endl;
            // K-way merge
            vector<tuwtype_t*> sources(s_k + 1);
            vector<size_t> lengths(s_k + 1);
            sources[0] = curr;
            lengths[0] = merged_len;
            for (size_t i = 0; i < s_k; ++i) {
                sources[i + 1] = static_cast<tuwtype_t*>(recvbuf) + ((1 << k) + i) * sendcount;
                lengths[i + 1] = sendcount;
            }
            kway_merge_variable_lengths(sources.data(), lengths.data(), s_k + 1, next);
            curr_len = merged_len + unmerged_len;
        }
        // Puffer für nächste Runde tauschen
        std::swap(curr, next);
    }

    // Ergebnis zurück in recvbuf
    std::copy(curr, curr + curr_len, static_cast<tuwtype_t*>(recvbuf));
    return MPI_SUCCESS;
}
