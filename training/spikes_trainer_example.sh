#!/usr/bin/env bash

python spectral-phase-retrieval/training/spikes_trainer.py run_pr_spikes_img --out_save_dir s3://phase-retrieval/spkies-recon/ae-conv-pred-range_2_8_lambda-embd-n_spikes_lambda01_lr5e-4_fft_none_zero_shift_img_norm_lambda_20 --n_iters_tr 20000 --spikes_range "[2, 8]" --epochs 35 --noised_mag False --is_proj_mag False --tile_size None --pred_type conv_ae --dbg_mode True