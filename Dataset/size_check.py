import numpy as np
from scipy import misc
file_name = '/home/aiswarya/Columbia_WoRk/OcclusionInference/Data/Hid_traverse_3dgt_border2/digts/gnrl/orig/orig_0.png'
face = misc.face()
misc.imsave(file_name, face)

print(face.shape)
