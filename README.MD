# High-Performance NTTs in CUDA

## Setup

```
 NTT_CUDA/
    ├── Makefile
    ├── include/ 
    └── src/ 
```

## Explanation for our code file:

<code>Ntt.cu</code>: optimized parallel CUDA code, including bit reversal removement, offline phase modular exponentiation, multi-thread computing and input-batching

<code>Ntt_as_is.cu</code>: first task, implement NTT in CUDA using single thread to compute

<code>NTT_step_a.cu</code>: a) bit reversal removement only

<code>NTT_step_b.cu</code>: adding offline phase modular exponentiation 

<code>NTT_step_b2.cu</code>: adding multi-thread computing

<code>NTT.cu</code>: adding input batching

<code>NTT_step_c.cu</code>: input batching testing

block size may affect the proper execution of the code

TODO: whether to change the block size from 1024 to 768 or 512 since it won't affect performance

TODO: currently only support Windows, need further optimization for Linux system
