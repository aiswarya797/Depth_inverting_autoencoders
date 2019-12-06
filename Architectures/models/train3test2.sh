python main_mod.py --train True --ckpt_name last --testing_method unsupervised --AE True \
     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 24 --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Data/Hid_traverse_3dgt_border2/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/Experiments/AE/Unfrozen/3_digts/BLT_BLT_depth_zdim24_2 \
