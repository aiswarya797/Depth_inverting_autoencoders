python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BL --decoder BL --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/occlusive/100k_BW_2digit_data_nrep4_z24_BLBL_cv2/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BL --decoder BT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/occlusive/100k_BW_2digit_data_nrep4_z24_BLBT_cv2/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BL --decoder BLT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/occlusive/100k_BW_2digit_data_nrep4_z24_BLBLT_cv2/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BT --decoder BT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/occlusive/100k_BW_2digit_data_nrep4_z24_BTBT_cv2/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BT --decoder BL --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/occlusive/100k_BW_2digit_data_nrep4_z24_BTBL_cv2/randomocc \

python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder BT --decoder BLT --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/100k_BW_2digit_data/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/occlusive/100k_BW_2digit_data_nrep4_z24_BTBLT_cv2/randomocc \


