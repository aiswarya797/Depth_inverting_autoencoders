
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

def reparametrize_gaussian(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def reparametrize_bernoulli(p_dist):
    eps = Variable(p_dist.data.new(p_dist.size()).uniform_(0,1))
    z = F.sigmoid(torch.log(eps + 1e-20) - torch.log(1-eps+ 1e-20) + torch.log(p_dist + 1e-20) - torch.log(1-p_dist+ 1e-20))
    return z


##### UNSUPERVISED ######
class shallow_autoencoder(torch.nn.Module):
	def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
		super(shallow_autoencoder, self).__init__() #error without this line#AttributeError: cannot assign module before Module.__init__() call
		self.nrep = n_rep
		self.nc = n_filter*32
		self.z_dim_tot = z_dim_bern + z_dim_gauss


		## Encoder part
		#self.fc1 = nn.Linear(self.nc, 256)
		self.fc1 = nn.Linear(self.nc, z_dim_bern)
		self.rc1 = nn.Linear(z_dim_bern,self.nc)
		#self.rc1 = nn.Linear(24,self.nc)
		
		## Decoder part
		self.fc2 = nn.Linear(z_dim_bern,self.nc)
		#self.fc4 = nn.Linear(256,self.nc)
		self.rc2 = nn.Linear(self.nc,z_dim_bern)
		#self.rc4 = nn.Linear(self.nc, 256)
		
	def forward(self, X, current_flip_idx_norm=None, train=True):
		final_z = []
		
		for t in range(self.nrep):
			if t<1:
				z1 = X
				z2 = self.fc1(z1)
				z3 = self.fc2(z2)
				final_z.append(z3)
			elif t>=1:
				rec1 = self.rc1(z2)
				z1 = X + rec1
				z1 = X
				rec2 = self.rc2(z3)
				z2 = self.fc1(z1)+rec2
				z3 = self.fc2(z2)
				final_z.append(z3)
		

		p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
		return final_z,p,mu,logvar,z3

# shallow_autoencoder_with_lateral two fully connected layers. Included top down, lateral and bottom up connections.
class shallow_autoencoder_with_lateral(torch.nn.Module):
	def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
		super(shallow_autoencoder_with_lateral, self).__init__() #error without this line#AttributeError: cannot assign module before Module.__init__() call
		self.nrep = n_rep
		self.nc = 1024
		self.z_dim_tot = z_dim_bern + z_dim_gauss

		## Encoder part
		self.fc1 = nn.Linear(self.nc, 24)
		self.rc1 = nn.Linear(24,self.nc)
		
		## Lateral connection at code layer
		self.lc1 = nn.Linear(24,24)
		
		## Decoder part
		self.fc2 = nn.Linear(24,self.nc)
		self.rc2 = nn.Linear(self.nc,24)
		
	def forward(self, X, current_flip_idx_norm=None, train=True):
		
		final_z = []
		for t in range(self.nrep):
			if t<1:
				z1 = X	#1024
				z2 = self.fc1(z1)	#24
				z3 = self.lc1(z2)	#24
				z4 = self.fc2(z3)	#1024
				final_z.append(z4)
			elif t>=1:
				rec1 = self.rc1(z2)	#1024
				z1 = X + rec1	#1024
				z2 = self.fc1(z1)	#24
				rec2 = self.rc2(z4)	#24
				z3 = self.lc1(z2)+rec2	#24
				z4 = self.fc2(z3)	#1024
				final_z.append(z4)
		p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
		return final_z,p,mu,logvar,z4		
		
	
# _3linear_autoencoder_with_lateral 6 fully connected layers. Included top down, lateral and bottom up connections.
class _3linear_autoencoder(torch.nn.Module):
	def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
		super(_3linear_autoencoder, self).__init__() #error without this line#AttributeError: cannot assign module before Module.__init__() call
		self.nrep = n_rep
		self.nc = 1024
		self.z_dim_tot = z_dim_bern + z_dim_gauss

		## Encoder part
		self.fc1 = nn.Linear(self.nc, 512)
		self.rc1 = nn.Linear(512,self.nc)
		self.fc2 = nn.Linear(512, 256)
		self.rc2 = nn.Linear(256,512)
		self.fc3 = nn.Linear(256, 24)
		self.rc3 = nn.Linear(24,256)
		
		## Lateral connection at code layer
		self.lc1 = nn.Linear(24,24)
		
		## Decoder part
		self.fc4 = nn.Linear(24,256)
		self.rc4 = nn.Linear(256,24)
		self.fc5 = nn.Linear(256,512)
		self.rc5 = nn.Linear(512,256)
		self.fc6 = nn.Linear(512, self.nc)
		self.rc6 = nn.Linear(self.nc, 512)
		
	def forward(self, X, current_flip_idx_norm=None, train=True):
		final_z = []
		for t in range(self.nrep):
			if t<1:
				z1 = X	#1024
				z2 = self.fc1(z1)	
				z3 = self.fc2(z2)
				z4 = self.fc3(z3)
				z5 = self.lc1(z4) 
				z6 = self.fc4(z5)
				z7 = self.fc5(z6)
				z8 = self.fc6(z7)
				final_z.append(z8)
			elif t>=1:
				rec1 = self.rc1(z2)	
				z1 = X + rec1	
				rec2 = self.rc2(z3)
				z2 = self.fc1(z1) + rec2
				rec3 = self.rc3(z4)
				z3 = self.fc2(z2) + rec3
				z4 = self.fc3(z3)
				rec4 = self.rc4(z6)
				z5 = self.lc1(z4) + rec4
				rec5 = self.rc5(z7)
				z6 = self.fc4(z5) + rec5
				rec6 = self.rc6(z8)
				z7 = self.fc5(z6) + rec6
				z8 = self.fc6(z7)
				
				final_z.append(z8)
		p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
		return final_z,p,mu,logvar,z8	
class _3linear_autoencoder_lateral(torch.nn.Module):
	def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
		super(_3linear_autoencoder_lateral, self).__init__() #error without this line#AttributeError: cannot assign module before Module.__init__() call
		self.nrep = n_rep
		self.nc = 1024
		self.z_dim_tot = z_dim_bern + z_dim_gauss

		## Encoder part
		self.fc1 = nn.Linear(self.nc, 512)
		self.rc1 = nn.Linear(512,self.nc)
		self.fc2 = nn.Linear(512, 256)
		self.rc2 = nn.Linear(256,512)
		self.fc3 = nn.Linear(256, 24)
		self.rc3 = nn.Linear(24,256)
		
		## Lateral connection at code layer
		self.lc1 = nn.Linear(512,512)
		self.lc2 = nn.Linear(256,256)
		self.lc3 = nn.Linear(24,24)
		self.lc4 = nn.Linear(256,256)
		self.lc5 = nn.Linear(512,512)
		
		## Decoder part
		self.fc4 = nn.Linear(24,256)
		self.rc4 = nn.Linear(256,24)
		self.fc5 = nn.Linear(256,512)
		self.rc5 = nn.Linear(512,256)
		self.fc6 = nn.Linear(512, self.nc)
		self.rc6 = nn.Linear(self.nc, 512)
		
	def forward(self, X, current_flip_idx_norm=None, train=True):
		final_z = []
		for t in range(self.nrep):
			if t<1:
				z1 = X	#1024
				z2 = self.fc1(z1)	
				z3 = self.lc1(z2)
				z4 = self.fc2(z3)
				z5 = self.lc2(z4)
				z6 = self.fc3(z5)
				z7 = self.lc3(z6)
				z8 = self.fc4(z7)
				z9 = self.lc4(z8)
				z10 = self.fc5(z9)
				z11 = self.lc5(z10)
				z12 = self.fc6(z11)
				final_z.append(z12)
			elif t>=1:
				r1 = self.rc1(z3)
				r2 = self.rc2(z5)
				r3 = self.rc3(z7)
				r4 = self.rc4(z9)
				r5 = self.rc5(z11)
				r6 = self.rc6(z12)
				z1 = X + r1	#1024
				z2 = self.fc1(z1)	
				z3 = self.lc1(z2)+r2
				z4 = self.fc2(z3)
				z5 = self.lc2(z4)+r3
				z6 = self.fc3(z5)
				z7 = self.lc3(z6)+r4
				z8 = self.fc4(z7)
				z9 = self.lc4(z8)+r5
				z10 = self.fc5(z9)
				z11 = self.lc5(z10)+r6
				z12 = self.fc6(z11)
				final_z.append(z12)
				
		p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
		return final_z,p,mu,logvar,z12

class syn_encoder_decoder(nn.Module):
	def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
		super (syn_encoder_decoder, self).__init__()
		self.encoder_type = encoder_type
		self.decoder_type = decoder_type
		self.n_rep = n_rep
		self.n_filt = n_filter
		self.z_dim_bern = z_dim_bern
		self.z_dim_gauss = z_dim_gauss
		self.z_dim_tot = z_dim_bern + z_dim_gauss
		self.nc = nc
		self.n_filter = n_filter
		self.sbd = sbd
		self.AE = AE

		self.W_b_1_e = nn.Conv2d(nc, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)   # bs 32 16 16
		self.W_b_2_e = nn.Conv2d(n_filter, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)
		self.W_b_3_e = nn.Conv2d(n_filter, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)
	    
		if encoder_type == 'BL' or encoder_type == 'BLT':
		    self.W_l_1_e = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
		    self.W_l_2_e = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
		    self.W_l_3_e = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
		if encoder_type == 'BT' or encoder_type == 'BLT':
		    self.W_t_1_e = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=4, stride=2, padding=1, output_padding=0 ,bias=False )
		    self.W_t_2_e = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=4, stride=2, padding=1, output_padding=0 ,bias=False )
	    

		self.Lin_1_e = nn.Linear(n_filter*4*4, 256, bias=True)
		self.Lin_2_e = nn.Linear(256, 256, bias=True)
		self.Lin_3_e = nn.Linear(256, (z_dim_bern+2*z_dim_gauss), bias=True)

		self.LRNe = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)

		self.Lin_1_d = nn.Linear( z_dim_bern + z_dim_gauss, 256, bias=True)
		self.Lin_2_d = nn.Linear(256, 256, bias=True) 
		self.Lin_3_d = nn.Linear(256, n_filter*4*4, bias=True)

		self.W_b_1_d = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=k, stride=2, padding=p, bias=True )
		self.W_b_2_d = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=k, stride=2, padding=p, bias=True )
		self.W_b_3_d = nn.ConvTranspose2d(n_filter, nc, kernel_size=k, stride=2, padding=p, bias=True ) 
	    
		if decoder_type == 'BL' or decoder_type == 'BLT':
			self.W_l_1_d = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
			self.W_l_2_d = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
			self.W_l_3_d = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
		if decoder_type == 'BT' or decoder_type == 'BLT':
			self.W_t_1_d = nn.Conv2d(n_filter, n_filter, kernel_size= 4, stride = 2, padding = 1, bias=False)
			self.W_t_2_d = nn.Conv2d(n_filter, n_filter, kernel_size= 4, stride = 2, padding = 1, bias=False) 

		    
		self.LRNd = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
	    
	def forward(self, x, train = True):
	    final_z_list = []
	    
	    if self.encoder_type == 'B':
			Z_1 = self.W_b_1_e(x)
			Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1)))
			Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2)))
			z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
			Z_4 = self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4)
			Z_5 = self.W_b_1_d(F.relu(Z_4))
			Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5)))
			final_z_list.append(self.W_b_3_d(self.LRNd(F.relu(Z_6))))	

	    elif self.encoder_type == 'BL':
	        for t in range(self.n_rep):
	            if t <1:
					Z_1 = self.W_b_1_e(x)
					Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1)))
					Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2)))
					z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
					Z_4 = self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4)
					Z_5 = self.W_b_1_d(F.relu(Z_4))
					Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5)))
					final_z_list.append(self.W_b_3_d(self.LRNd(F.relu(Z_6)))) 
				     
	            elif t>=1:
	                Z_1 = self.W_b_1_e(x) + self.W_l_1_e(self.LRNe(F.relu(Z_1)))
	                Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1))) + self.W_l_2_e(self.LRNe(F.relu(Z_2))) 
	                Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2))) + self.W_l_3_e(self.LRNe(F.relu(Z_3)))
	                z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRNe(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
	                Z_4 =  self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4) + self.W_l_1_d(self.LRNd(F.relu(Z_4))) 
	                Z_5 = self.W_b_1_d(self.LRNd(F.relu(Z_4))) + self.W_l_2_d(self.LRNd(F.relu(Z_5))) 
	                Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5))) + self.W_l_3_d(self.LRNd(F.relu(Z_6)))
	                final_z_list.append(self.W_b_3_d(self.LRNd(F.relu(Z_6))))
	                
	                
	    elif self.encoder_type == 'BT':
	        for t in range(self.n_rep):
	            if t <1:
				    Z_1 = self.W_b_1_e(x)
				    Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1)))
				    Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2)))
				    z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
				    Z_4 = self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4)
				    Z_5 = self.W_b_1_d(F.relu(Z_4))
				    Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5)))
				    final_z_list.append(self.W_b_3_d(self.LRNd(F.relu(Z_6)))) 
	            elif t>=1:
	                Z_1 = self.W_b_1_e(x) + self.W_t_1_e(self.LRNe(F.relu(Z_2))) 
	                Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1))) + self.W_t_2_e(self.LRNe(F.relu(Z_3)))
	                Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2)))
	                z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRNe(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
	                Z_4 =  self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4) + self.W_t_1_d(self.LRNd(F.relu(Z_5))) 
	                Z_5 = self.W_b_1_d(self.LRNd(F.relu(Z_4))) +  self.W_t_2_d(self.LRNd(F.relu(Z_6))) 
	                Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5))) 
	                final_z_list.append(self.W_b_3_d(self.LRN(F.relu(Z_6))))
	                
	    elif self.encoder_type == 'BLT':
	        for t in range(self.n_rep):
	            if t <1:
					Z_1 = self.W_b_1_e(x)
					Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1)))
					Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2)))
					z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRNe(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
					Z_4 = self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4)
					Z_5 = self.W_b_1_d(F.relu(Z_4))
					Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5)))
					final_z_list.append(self.W_b_3_d(self.LRNd(F.relu(Z_6)))) 
				    
	            elif t>=1:
					Z_1 = self.W_b_1_e(x) + self.W_l_1_e(self.LRNe(F.relu(Z_1))) + self.W_t_1_e(self.LRNe(F.relu(Z_2))) 
					Z_2 = self.W_b_2_e(self.LRNe(F.relu(Z_1))) + self.W_l_2_e(self.LRNe(F.relu(Z_2))) + self.W_t_2_e(self.LRNe(F.relu(Z_3))) 
					Z_3 = self.W_b_3_e(self.LRNe(F.relu(Z_2))) + self.W_l_3_e(self.LRNe(F.relu(Z_3)))
					z = self.Lin_3_e(F.relu(self.Lin_2_e(F.relu(self.Lin_1_e(self.LRNe(F.relu(Z_3)).view(-1, self.n_filt*4*4 ))))))
					Z_4 =  self.Lin_3_d(F.relu(self.Lin_2_d(F.relu(self.Lin_1_d(z))))).view(-1,self.n_filter,4,4) + self.W_t_1_d(self.LRNd(F.relu(Z_5))) + self.W_l_1_d(self.LRNd(F.relu(Z_4))) 
					Z_5 = self.W_b_1_d(self.LRNd(F.relu(Z_4))) +  self.W_t_2_d(self.LRNd(F.relu(Z_6))) + self.W_l_2_d(self.LRNd(F.relu(Z_5))) 
					Z_6 = self.W_b_2_d(self.LRNd(F.relu(Z_5))) + self.W_l_3_d(self.LRNd(F.relu(Z_6)))
					final_z_list.append(self.W_b_3_d(self.LRNd(F.relu(Z_6))))

		p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
	    return final_z_list,p,mu,logvar,final_z_list[-1]

class multi_encoder(nn.Module):
    def __init__(self, encoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p):
        super (multi_encoder, self).__init__()
        self.encoder_type = encoder_type
        self.n_rep = n_rep
        self.n_filt = n_filter
        
        
        self.W_b_1 = nn.Conv2d(nc, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)   # bs 32 16 16
        self.W_b_2 = nn.Conv2d(n_filter, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)
        self.W_b_3 = nn.Conv2d(n_filter, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)
        
        if encoder_type == 'BL' or encoder_type == 'BLT':
            self.W_l_1 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_2 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_3 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
        if encoder_type == 'BT' or encoder_type == 'BLT':
            self.W_t_1 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=4, stride=2, padding=1, output_padding=0 ,bias=False )
            self.W_t_2 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=4, stride=2, padding=1, output_padding=0 ,bias=False )
        
        if encoder_type == 'Lin':
        	self.L1 = nn.Linear(n_filter*4*4*2, 512, bias=True) # n_filter*4*4,256,
        	self.L2 = nn.Linear(512, 256, bias=True)# 256,256
        	self.L3 = nn.Linear(256, (z_dim_bern+2*z_dim_gauss), bias=True)
        
        if encoder_type == 'Lin2':
        	self.L1 = nn.Linear(n_filter*4*4*2, 512, bias=True) # n_filter*4*4,256,
        	self.L2 = nn.Linear(512, 512,bias = True)
        	self.L3 = nn.Linear(512, 256, bias=True)# 256,256
        	self.L4 = nn.Linear(256,256, bias = True)
        	self.L5 = nn.Linear(256, (z_dim_bern+2*z_dim_gauss), bias=True)

        self.Lin_1 = nn.Linear(n_filter*4*4, 256, bias=True) # n_filter*4*4,256,
        self.Lin_2 = nn.Linear(256, 256, bias=True)# 256,256
        self.Lin_3 = nn.Linear(256, (z_dim_bern+2*z_dim_gauss), bias=True)
        
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        self.weight_init()
    
    def weight_init(self):
        for block in self._modules.values():
            #print(block)
            if isinstance(block, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(block.weight,  nonlinearity='relu')
                if block.bias is not None:
                    block.bias.data.fill_(0)
    
    def forward(self, x):
        final_z_list = []
        if self.encoder_type == 'Lin':
        	final_z_list.append(self.L3(F.relu(self.L2(F.relu(self.L1(F.relu(x)))))))

        elif self.encoder_type == 'Lin2':
        	final_z_list.append(self.L5(F.relu(self.L4(F.relu(self.L3(F.relu(self.L2(F.relu(self.L1(F.relu(x)))))))))))

        elif self.encoder_type == 'CL1':
        	Z_3 = self.W_b_1(F.relu(x))
        	final_z_list.append(self.Lin_3(F.relu(self.Lin_1(F.relu(Z_3).view(-1, self.n_filt*4*4 )))))

        elif self.encoder_type == 'B':
            Z_1 = self.W_b_1(x)
            Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
            Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))
            final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
            
        elif self.encoder_type == 'BL':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.W_b_1(x)
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) 
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
                elif t>=1:
                    Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1)))
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
        elif self.encoder_type == 'BT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.W_b_1(x)
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))   
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
                elif t>=1:
                    Z_1 = self.W_b_1(x) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_t_2(self.LRN(F.relu(Z_3)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
        elif self.encoder_type == 'BLT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.W_b_1(x)
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) 
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
                elif t>=1:
                    Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1))) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) + self.W_t_2(self.LRN(F.relu(Z_3))) 
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
            
        
        #final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
           
                
        #print(torch.sum(torch.isnan(final_z)))
        #print(final_z.size())
        #print(F.sigmoid(final_z[0,:]))
        return(final_z_list)

    
class multi_decoder(nn.Module):
    def __init__(self, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p):
        super (multi_decoder, self).__init__()    
        self.decoder_type = decoder_type
        self.n_rep = n_rep
        self.n_filter = n_filter
        
        self.Lin_1 = nn.Linear( z_dim_bern + z_dim_gauss, 256, bias=True)
        self.Lin_2 = nn.Linear(256, 256, bias=True) 
        self.Lin_3 = nn.Linear(256, n_filter*4*4, bias=True)
        
        self.W_b_1 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=k, stride=2, padding=p, bias=True )
        self.W_b_2 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=k, stride=2, padding=p, bias=True )
        self.W_b_3 = nn.ConvTranspose2d(n_filter, nc, kernel_size=k, stride=2, padding=p, bias=True ) 
        
        if decoder_type == 'BL' or decoder_type == 'BLT':
            self.W_l_1 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_2 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_3 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
        if decoder_type == 'BT' or decoder_type == 'BLT':
            self.W_t_1 = nn.Conv2d(n_filter, n_filter, kernel_size= 4, stride = 2, padding = 1, bias=False)
            self.W_t_2 = nn.Conv2d(n_filter, n_filter, kernel_size= 4, stride = 2, padding = 1, bias=False) 
            
        if decoder_type == 'Lin':
        	self.L3 = nn.Linear(512,n_filter*4*4*2, bias=True) # n_filter*4*4,256,
        	self.L2 = nn.Linear(256,512, bias=True)# 256,256
        	self.L1 = nn.Linear((z_dim_bern+2*z_dim_gauss),256, bias=True)

        if decoder_type == 'Lin2':
        	self.L5 = nn.Linear(512,n_filter*4*4*2, bias=True) # n_filter*4*4,256,
        	self.L4 = nn.Linear(512,512, bias = True)
        	self.L3 = nn.Linear(256,512, bias=True)# 256,256
        	self.L2 = nn.Linear(256,256, bias = True)
        	self.L1 = nn.Linear((z_dim_bern+2*z_dim_gauss),256, bias=True)

        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
      
        self.weight_init()
        
    def weight_init(self):
        for block in self._modules.values():
            #print(block)
            if isinstance(block, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(block.weight,  nonlinearity='relu')
                if block.bias is not None:
                    block.bias.data.fill_(0)
            
                    
    def forward(self, z):
        final_img_list = []
        if self.decoder_type == 'Lin':
        	a = self.L3(F.relu(self.L2(F.relu(self.L1(z)))))
        	final_img_list.append(a)

        elif self.decoder_type == 'Lin2':
        	final_img_list.append(self.L5(F.relu(self.L4(F.relu(self.L3(F.relu(self.L2(F.relu(self.L1(z))))))))))
        	

        elif self.decoder_type == 'CL1':
        	Z_1 = self.Lin_3(F.relu(self.Lin_1(z))).view(-1, self.n_filter,4,4)
        	a = self.W_b_3(self.LRN(F.relu(Z_1)))
        	print(a.type())
        	final_img_list.append(a)

        elif self.decoder_type == 'B':
            Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
            Z_2 = self.W_b_1(F.relu(Z_1))
            Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
            final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
        
        elif self.decoder_type == 'BL':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
                    Z_2 = self.W_b_1(F.relu(Z_1))
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
                if t>=1:
                    Z_1 =  self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4) + self.W_l_1(self.LRN(F.relu(Z_1))) 
                    Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
        elif self.decoder_type =='BT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
                    Z_2 = self.W_b_1(F.relu(Z_1))
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
                if t>=1:
                    Z_1 =  self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                    Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) +  self.W_t_2(self.LRN(F.relu(Z_3))) 
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) 
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
        elif self.decoder_type == 'BLT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
                    Z_2 = self.W_b_1(F.relu(Z_1))
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
                if t>=1:
                    Z_1 =  self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4) + self.W_t_1(self.LRN(F.relu(Z_2))) + self.W_l_1(self.LRN(F.relu(Z_1))) 
                    Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) +  self.W_t_2(self.LRN(F.relu(Z_3))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
                
        #final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
        #final_img = self.W_b_3(self.LRN(F.relu(Z_3)))
        return(final_img_list)
        
        
class multi_VAE(nn.Module):
    def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
        super (multi_VAE, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.z1_file = '/home/aiswarya/Columbia_WoRk/z1.pt'
        self.z_dim_bern = z_dim_bern
        self.z_dim_gauss = z_dim_gauss
        self.z_dim_tot = z_dim_bern + z_dim_gauss
        self.nc = nc
        self.n_filter = n_filter
        self.sbd = sbd
        self.AE = AE
        if self.AE:
            print("NORMAL AUTOENCODER")
        
        self.encoder = multi_encoder(encoder_type,z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p)
        if sbd == True:
            self.decoder = SB_decoder(z_dim_bern, z_dim_gauss, n_filter, nc)
            self.sbd_model = spatial_broadcast_decoder()
            print('using {} encoder SBD decoder'.format(encoder_type))
        else:
            self.decoder = multi_decoder(decoder_type,z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p)
            print('using {} encoder {} decoder'.format(encoder_type, decoder_type))

    def forward(self, x, current_flip_idx_norm=None, train=True, z_anal = False):
        if train==True:
            if not self.AE:
                distributions = self._encode(x)
                distributions = distributions[-1]
                if self.z_dim_bern == 0:
                    p = 0
                    mu = distributions[:,:self.z_dim_gauss]
                    logvar = distributions[:, self.z_dim_gauss: ]
                    z = reparametrize_gaussian(mu, logvar)
                elif self.z_dim_gauss == 0:
                    
                    p = F.sigmoid(p)
                    if current_flip_idx_norm is not None:
                        delta_mat = torch.zeros(p.size())
                        delta_mat[current_flip_idx_norm,1]=1
                        p = p - 2*delta_mat*p + delta_mat
                    z =  reparametrize_bernoulli(p)
                elif self.z_dim_bern !=0 and self.z_dim_gauss != 0:
                    p = distributions[:, :self.z_dim_bern]
                    p = F.sigmoid(p)
                    if current_flip_idx_norm is not None:
                        delta_mat = torch.zeros(p.size())
                        delta_mat[current_flip_idx_norm,1]=1
                        p = p - 2*delta_mat*p + delta_mat
                    mu = distributions[:,self.z_dim_bern:(self.z_dim_bern+self.z_dim_gauss) ]
                    logvar = distributions[:, (self.z_dim_bern+self.z_dim_gauss):]
                    bern_z =reparametrize_bernoulli(p)
                    gauss_z = reparametrize_gaussian(mu, logvar)
                    z = torch.cat((bern_z,gauss_z), 1)

                if self.sbd:
                    z = self.sbd_model(z)
            elif self.AE:
                
                z = self._encode(x)

                #print("z_max", torch.max(z[-1]))
                #print("z_min", torch.min(z[-1]))
                
                p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
                #print("No reparam!")
                
            x_recon_list = self._decode(z[-1])
            #x_recon_list = [m.view(x.size()) for m in x_recon_list]
            return x_recon_list, p, mu, logvar, z[-1]
        
        elif train ==False and z_anal == False:
            distributions_list = self._encode(x)
            distributions = distributions_list[-1]
            z = distributions
            #p = distributions[:, :self.z_dim_bern]
            #mu = distributions[:,self.z_dim_bern:(self.z_dim_bern+self.z_dim_gauss) ]
            #z = torch.cat((p,mu), 1)
            if self.sbd:
                z = self.sbd_model(z)
            x_recon = self._decode(z)
            #print(x_recon.shape)
            return x_recon

        elif z_anal == True:
			print('z_anal')
			distributions_list = self._encode(x)
			z = distributions_list[-1] #shape (batch_size, code layer size)
			
			z_new = torch.load('/home/aiswarya/z_tensor_list_mean.pt')
			z_new = z_new + z

			x_recon = self._decode(z_new)
			return x_recon
			

    
    def _encode(self,x):
        return(self.encoder(x))
    
    def _decode(self,z):
        return(self.decoder(z))  

     
    
class SB_decoder(nn.Module):
    def __init__(self, z_dim_bern, z_dim_gauss, n_filter, nc):
        super(SB_decoder, self).__init__()  
            
        self.decoder = nn.Sequential(
            nn.Conv2d((z_dim_bern + z_dim_gauss + 2), n_filter, 3, 1, 1),     
            nn.ReLU(True),
            nn.Conv2d(n_filter, n_filter, 3, 1, 1),         
            nn.ReLU(True),
            nn.Conv2d(n_filter, nc, 3, 1, 1),         
        )
        self.weight_init() 
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        recon = self._decode(z)
        return(recon)
       
    def _decode(self,z):
        return(self.decoder(z))

    
class spatial_broadcast_decoder(nn.Module):
    def __init__(self, im_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(spatial_broadcast_decoder, self).__init__()
        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.to(self.device)
        y_grid = y_grid.to(self.device)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
    
    def forward(self,z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))
        # Tile across to match image size
        # Shape: NxDx32x32
        z = z.expand(-1, -1, self.im_size, self.im_size)
        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x32x32
        z_bd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        return(z_bd)
    
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

    
    
    
    
    
    
    
    
###### SUPERVISED ######

            
class BLT_orig_encoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_orig_encoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using BLT_orig_encoder")
        
        self.W_b_1 = nn.Conv2d(1, 32, kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.W_l_1 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=False )
        self.W_b_2 = nn.Conv2d(32, 32, kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.W_l_2 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.Lin = nn.Linear(32, self.z_dim, bias=True)
        
        self.MPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        
        self.weight_init()
        
    def weight_init(self):
        for block in self._modules.values():
            if isinstance(block, (nn.Linear, nn.Conv2d)):
                print(block)
                init.kaiming_normal(block.weight,  nonlinearity='relu')
                if block.bias is not None:
                    block.bias.data.fill_(0)

    def forward(self, x):
        for t in range(4):
            if t<1:
                Z_1 = self.W_b_1(x)
                Z_2_mpool, indices_hid  = self.MPool(Z_1)
                Z_2 = self.W_b_2(self.LRN(F.relu(Z_2_mpool)))
                read_out, indices_max =  F.max_pool2d_with_indices(self.LRN(F.relu(Z_2)), kernel_size=Z_2.size()[2:],
                                                               return_indices=True )
                final_z = self.Lin(read_out.view(-1, 32))
            if t >=1:
                Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1))) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                Z_2_mpool, indices_hid  = self.MPool(Z_1)
                Z_2 = self.W_b_2(self.LRN(F.relu(Z_2_mpool))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                read_out, indices_max =  F.max_pool2d_with_indices(self.LRN(F.relu(Z_2)), kernel_size=Z_2.size()[2:],
                                                               return_indices=True )
                final_z = self.Lin(read_out.view(-1, 32))
    
        #print(torch.sum(torch.isnan(final_z)))
        #print(F.sigmoid(final_z[0,:]))
        #print(final_z.size())
        return(final_z)

class BLT_orig(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_orig, self).__init__()
        self.encoder = BLT_orig_encoder(z_dim, nc)
        
    def forward(self, x):
        z = self._encode(x)
        #recon = decoder(z)
        return(z)
    
    def _encode(self,x):
        return(self.encoder(x))
    

    