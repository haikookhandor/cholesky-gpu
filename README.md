# Parallel Cholesky Factorization on GPU (CUDA)

CUDA implementation of Cholesky decomposition for **SPD** (symmetric positive-definite) matrices, with a focus on performance tuning via **thread-block configuration** and **memory-access optimization**.

This project includes:
- A CPU baseline + correctness checks
- A GPU implementation for Cholesky factorization

> Benchmarks for the results were run on NVIDIA GPUs in an HPC environment (Slurm), but this repository keeps the code self-contained and runnable on any CUDA-capable machine.

---

## Highlights

- Cholesky decomposition for SPD matrices on NVIDIA GPUs using CUDA
- Designed to explore performance across thread/block/grid configurations
- Validates numerical correctness against a CPU reference implementation
- Includes timing/benchmark printing for scalability experiments


## Project structure
- src/cholesky_gpu_single.cu # CPU baseline
- src/cholesky_gpu_parallel.cu # tuned/parallel GPU implementation


## Run

Both programs take the following arguments:

```bash
./cholesky_parallel <n> <tpb> <block_x> <block_y>
```

Where:
- `n` = matrix dimension (n x n)
- `tpb` = threads-per-block (1D parameter used in parts of the implementation)
- `block_x`, `block_y` = 2D block dimensions used for kernel configuration

Example:

```bash
./build/cholesky_parallel 1024 256 16 16
```

## Benchmarking 

You can sweep matrix sizes and configurations using a simple loop:

```bash
for n in 512 1024 2048; do
  for tpb in 128 256 512; do
    ./build/cholesky_parallel $n $tpb 16 16
  done
done
```

