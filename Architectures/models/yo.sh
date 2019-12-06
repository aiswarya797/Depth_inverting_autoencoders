python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BL --decoder BLT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLBLT/randomocc \


python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BT --decoder BLT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BTBLT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder B --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLTB/randomocc \


python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder B --decoder BLT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BBLT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder B --decoder B --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BB/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BT --decoder BL --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BTBL/randomocc \


python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BL --decoder BT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLBT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder BLT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLTBLT/randomocc \


python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BT --decoder BT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BTBT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BL --decoder BL --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLBL/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder BT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLTBT/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BLT --decoder BL --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/75k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/75K2digit/75k_100_100BW_2digits_nrep4_z24_BLTBL/randomocc \
