Explanation for our code file:
Ntt.cu: optimized parallel CUDA code, including bit reversal removement, offline phase modular exponentiation, multi-thread computing and input-batching
Ntt _as_is.cu: first task, implement NTT in CUDA using single thread to compute
NTT _step_a.cu: a) bit reversal removement only
NTT _step_b.cu: adding offline phase modular exponentiation 
NTT _step_b2.cu: adding multi-thread computing
NTT.cu: adding input batching
NTT_step_c.cu: input batching testing

block size may affect the proper execution of the code
TODO: whether to change the size from 1024 to 768 or 512 since it won't affect performance
TODO: currently only support Windows, need further optimization for Linux system
