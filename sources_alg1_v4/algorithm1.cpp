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
    int origin; // Process rank that sent this value

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

    // Nur ein temporärer Puffer für Merging - der andere ist recvbuf
    vector<TrackedElement> buffer_B(p * sendcount);
    
    // Temporärer Puffer für die Arbeit mit recvbuf
    vector<TrackedElement> recvbuf_tracked(p * sendcount);
    
    //Two temporary buffers for communication
    vector<TrackedElement> sendbuffer(p * sendcount);
    vector<TrackedElement> recvbuffer(p * sendcount);

    //Initialize tracked recvbuf with local data and own process rank
    const tuwtype_t* local_data = static_cast<const tuwtype_t*>(sendbuf);
    for (int i = 0; i < sendcount; ++i) {
        recvbuf_tracked[i].value = local_data[i];
        recvbuf_tracked[i].origin = r; // Store the origin process rank
    }

    TrackedElement* curr = recvbuf_tracked.data();
    TrackedElement* next = buffer_B.data();
    int curr_len = sendcount;
    int s_k, t, f;
    
    // Neue Hilfsvariablen
    TrackedElement* sendptr;
    int send_len;

    //Bruck algorithm with 2-way merges and tracking
    for (int k = 0; k < q; ++k) {
        s_k = 1 << k;
        t = (r - s_k + p) % p; // Target process to send to
        f = (r + s_k) % p; // Source process to receive from

        /* ---------- Bestimmen, was wir verschicken ---------- */
        if (k == q-1) {
            s_k -= ((1 << q) - p); // Adjust for last round
            
            send_len = 0;
            for (int i = 0; i < curr_len; ++i) {
                if (int(curr[i].origin - r + p) % p < s_k) {
                    sendbuffer[send_len++] = curr[i];  // selektiv kopieren
                }
            }
            sendptr = sendbuffer.data();  // gefilterter Puffer
        } else {
            sendptr = curr;              // >>> NEU: direkt aus curr senden
            send_len = curr_len;
        }

        /* ---------- Kommunikation ---------- */
        MPI_Sendrecv(
            sendptr, send_len * sizeof(TrackedElement), MPI_BYTE, t, 0,
            recvbuffer.data(), send_len * sizeof(TrackedElement), MPI_BYTE, f, 0,
            comm, MPI_STATUS_IGNORE
        );
        
        /* ---------- Merge & Vorbereiten für nächste Runde ---------- */
        merge_with_tracking(curr, curr_len, recvbuffer.data(), send_len, next);
        curr_len += send_len;

        // Swap buffers for next iteration
        std::swap(curr, next);
    }
    
    // Kopiere nur die Werte (nicht die Herkunftsinformationen) in recvbuf
    tuwtype_t* result = static_cast<tuwtype_t*>(recvbuf);
    for (int i = 0; i < curr_len; ++i) {
        result[i] = curr[i].value;
    }
    return MPI_SUCCESS;
}