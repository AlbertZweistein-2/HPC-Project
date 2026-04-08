# HPC Project

MPI-based implementations of the **allgather-merge** problem for the TU Wien *High Performance Computing* course. The project studies how different communication and merging strategies affect scalability when each MPI rank starts with a locally sorted block and all ranks must end up with the full globally sorted sequence. :contentReference[oaicite:0]{index=0}

## Overview

The task is:

> Given $p$ processes, each holding a sorted block of size $m/p$, compute on every process the full sorted array of size $m$ without losing or introducing duplicates. :contentReference[oaicite:1]{index=1}

This repository contains three MPI/C++ approaches:

- **Baseline**: `MPI_Allgather` followed by a local sequential $p$-way merge.
- **Bruck-based allgather-merge**: integrates communication rounds using a Bruck-style circulant communication pattern.
- **Circulant allreduce-style merge**: replaces reduction with merge operations and relies on repeated two-way merges of sorted data. :contentReference[oaicite:2]{index=2}

The main focus of the project is to compare these approaches with respect to:

- communication overhead
- merge complexity
- memory movement
- weak-scaling behavior on a cluster environment :contentReference[oaicite:3]{index=3}

## Implemented Algorithms

### 1. Baseline

The baseline solution uses `MPI_Allgather` to replicate all local blocks on every rank and then performs a local $p$-way merge using a min-heap / priority queue. This is simple and correct, but it becomes expensive because each rank performs the full merge independently. :contentReference[oaicite:4]{index=4}

**Idea**
- gather all sorted blocks
- merge all $p$ blocks locally
- write the globally sorted result to the output buffer

### 2. Bruck Allgather-Merge

The Bruck variant uses a straight-doubling circulant communication pattern. During development, several versions were explored, gradually reducing the reliance on expensive $p$-way merges. The final version exchanges only sorted data and performs only two-way merges, while tracking the origin rank of elements to support the final selective exchange. :contentReference[oaicite:5]{index=5}

**Key ideas**
- logarithmic communication rounds
- incremental merging during communication
- final version avoids costly global $p$-way merging
- origin tracking enables selective data exchange in the last round

### 3. Circulant Allreduce-Style Merge

This version follows a circulant allreduce communication scheme, but replaces reduction with merge operations. It performs repeated two-way merges of sorted buffers and reduces merge fan-in compared to the baseline. In the benchmarks, this was the strongest approach overall. :contentReference[oaicite:6]{index=6}

**Key ideas**
- $O(\log p)$ communication rounds
- repeated two-way merges instead of one large $p$-way merge
- lower merge overhead for large process counts

## Results Summary

Benchmarks were run on the **Hydra** cluster using weak scaling with:

- node counts: 1, 10, 20
- processes per node: 1, 10, 32
- local message sizes: $1, 10, 100, 1000, 10000, 100000$ elements per process
- 2 warm-up runs and 10 measured runs per configuration :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

### Main takeaway

The **Circulant** implementation showed the best weak-scaling behavior, especially once inter-node communication became dominant. The **Baseline** version scaled worst due to redundant communication and expensive local $p$-way merges. The **Bruck** implementation improved significantly over the baseline, but still trailed the circulant approach in larger configurations. :contentReference[oaicite:9]{index=9}

In short:

- **Baseline** = simplest, but least scalable
- **Bruck** = better communication pattern, good improvement
- **Circulant** = best overall performance in this project

## Repository Structure

The repository contains at least the following top-level items:
- 'HPC_Report_Group2.pdf' – the final project report
- `sources/` – source files for the MPI implementations
- `report/` – report material
- `build/` – build artifacts
- `hpc_project.pdf` – project report
- `Ex3_pseudocode.md` – pseudocode notes
- `README.md` – project overview :contentReference[oaicite:10]{index=10}

## Tech Stack

- C++
- MPI
- CMake
- SLURM for cluster jobs
- HPCToolkit / HPCViewer for profiling on Hydra