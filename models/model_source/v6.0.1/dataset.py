import torch
from torch.utils.data import Dataset
import h5py
import numpy as np 
from PIL import Image
import heatmap_generator
import matplotlib.pyplot as plt

class Dataset(Dataset):
  
  def __init__(self, h5file,std_filename,mean_filename):
    with h5py.File(h5file, 'r') as hdf:
      # Get the data      
      #print("landmarks")
      self.data_lm = np.asarray(hdf.get("landmarks"))
      #print("images")
      self.data_im = np.asarray(hdf.get("images"))
      #print("expressions")
      self.exprs = np.asarray(hdf.get("expressions"))
    hdf.close()
    
    # Load mean and std image
    #print(np.asarray(Image.open(mean_filename)).reshape(1,128,128).shape)
    self.mean_image = np.asarray(Image.open(mean_filename))/255#.transpose(2, 0, 1)
    self.std_image = np.asarray(Image.open(std_filename))/255#.transpose(2, 0, 1)
    #self.std_image = self.std_image.squeeze()

  def __len__(self):
    return len(self.exprs)


  def __getitem__(self, index):
    # Select sample
    #image = 0.21*self.data_im[index][0,:,:]+0.72*self.data_im[index][1,:,:]+0.07*self.data_im[index][2,:,:]
    #mean = 0.21*self.mean_image[0,:,:]+0.72*self.mean_image[1,:,:]+0.07*self.mean_image[2,:,:]
    #std = 0.21*self.std_image[0,:,:]+0.72*self.std_image[1,:,:]+0.07*self.std_image[2,:,:]
    #print(np.asarray([(image-mean)/std]).shape)
    #print("*******",self.data_lm[index].shape)
    return np.asarray((self.data_im[index]-self.mean_image)/self.std_image),np.asarray(heatmap_generator.getHeatMap(self.data_lm[index])), (self.exprs[index] if self.exprs[index] < 5 else 4)



