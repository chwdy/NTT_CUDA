#srun --gres=gpu:1 --pty /bin/bash
module load cuda/11.1.74
nvidia-smi
make -j8
./ntt