python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder BLT --n_rep 1 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/timestep/100k_BW_2digit_data_nrep1_z24_BLTBLT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder BLT --n_rep 2 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/timestep/100k_BW_2digit_data_nrep2_z24_BLTBLT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder BLT --n_rep 8 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/timestep/100k_BW_2digit_data_nrep8_z24_BLTBLT/randomocc \
