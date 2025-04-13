date

mpiexec -n 2 --hostfile hostfile \
        -outfile-pattern="rank-%r.out" \
	    python3 ~/moe/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py \
	      --input_folder /work/nvme/bcjw/xlian/ckpt/zero2/global_step5 \
	        --output_folder /work/nvme/bcjw/xlian/ckpt/zero2/global_step5_universal

date