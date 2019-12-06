"""main.py"""

import argparse

import numpy as np
import torch

import sys
#sys.path.insert(0, '/Users/riccardoconci/Desktop/code/ZuckermanProject/OcclusionInference/Architectures')
sys.path.insert(0, '/home/aiswarya/Columbia_WoRk/OcclusionInference/Architectures')

from solvers.unsup_solver import Solver_unsup
from solvers.sup_solver import Solver_sup
from solvers.utils_mod import str2bool
from solvers.visuals_mod import plotLearningCurves, linear_readout_sup

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
	print('columbiaWork')
	seed = args.seed
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

	if args.testing_method == 'supervised_encoder' or args.testing_method == 'supervised_decoder':
		print("SUPERVISED!")
		net = Solver_sup(args)
	elif args.testing_method == 'unsupervised':
		print("UNSUPERVISED!")
		net = Solver_unsup(args)	#here the self variables are assigned

	if args.train:
		print("Training")
		net.train()
		print("Testing")
		#net.gnrl_loss()
		if args.testing_method == 'unsupervised':
			net.test_plots()
			#linear_readout_sup(net, max_epoch = 50)
		print("plotting learning curves!")
		plotLearningCurves(net)


	elif not args.train:
		print("Testing")
		net.gnrl_loss()
		#if args.testing_method == 'unsupervised':
			#net.test_plots()
			#linear_readout_sup(net, max_epoch = 10)
		if args.testing_method == 'supervised_decoder':
			net.test_images()
		#print("plotting learning curves!")
		#plotLearningCurves(net)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='OcclusionInference')

	parser.add_argument('--train', default=True, type=str2bool, help='train or test')
	parser.add_argument('--seed', default=2019, type=int, help='random seed')
	parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')

	parser.add_argument('--encoder', default='B', type=str, help='B, BL, BT, BLT')
	parser.add_argument('--decoder', default='B', type=str, help='B, BL, BT, BLT')
	parser.add_argument('--sbd', default=False, type=str2bool, help='use spatial broacast decoder')
	parser.add_argument('--n_rep', default=4, type=int, help='iterations of recurrent processing ')
	parser.add_argument('--n_filter', default=32, type=int, help='number of filters in convolutional layers')
	parser.add_argument('--kernel_size', default=4, type=int, help='kernel size in convolutional layers')
	parser.add_argument('--padding', default=1, type=int, help='padding in convolutional layers')

	parser.add_argument('--AE', default=False, type=str2bool, help='use normal autoencoder without variational')
	parser.add_argument('--freeze_decoder', default=False, type=str2bool, help='freeze decoder of network')

	parser.add_argument('--z_dim_bern', default= 0, type=int, help='dimension of the representation z')
	parser.add_argument('--z_dim_gauss', default= 20, type=int, help='dimension of the representation z')
		
	parser.add_argument('--optim_type', default='Adam', type=str, help='type of optimiser')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--l2_loss', default=0.0, type=float, help='L2 loss coefficient')
	parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
	parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

	parser.add_argument('--max_epoch', default=60, type=float, help='maximum training epoch')
	parser.add_argument('--batch_size', default=128, type=int, help='batch size')
	parser.add_argument('--beta', default=1, type=float, help='beta parameter for KL-term in original beta-VAE')
	parser.add_argument('--gamma', default=1, type=float, help='gamma parameter for KL-term in bernoulli VAE')   

	parser.add_argument('--flip', default=False, type=str2bool, help='enable flipping of Zs during image inverse')

	parser.add_argument('--dset_dir', default='/train/', type=str, help='dataset directory')
	parser.add_argument('--dataset', default='digits_gray', type=str, help='dataset name')
	parser.add_argument('--testing_method', default='unsupervised', type=str, help='supervised vs unsupervised')
	parser.add_argument('--encoder_target_type', default='depth_black_white_xy_xy', type=str, help='types of supervised encoding')
	parser.add_argument('--image_size', default=32, type=int, help='image size. now only (64,64) is supported')
	parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')

	parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
	parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
	parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
	parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
	parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

	parser.add_argument('--gather_step', default=5, type=int, help='iteration for results gathering ')
	parser.add_argument('--display_step', default=10, type=int, help='iteration for data display')
	parser.add_argument('--save_step', default=600, type=int, help='iterations for checkpoint save')

	parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
	parser.add_argument('--ckpt_name', default='None', type=str, help='load previous checkpoint. insert checkpoint filename')

	args = parser.parse_args()

	main(args)
