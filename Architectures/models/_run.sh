python main_mod.py --train True --ckpt_name None --testing_method unsupervised --AE True \
     --encoder Lin2 --decoder Lin2 --n_rep 4 --freeze_decoder False --z_dim_bern 24  --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/aiswarya/Columbia_WoRk/blackwhitedigits_random/digts/digts/ \
     --output_dir /home/aiswarya/Columbia_WoRk/OcclusionInference/shallow_autoencoder_nonlinear/blackwhite_random/_5layer_noLT \
