import torch
from torch.utils.data import Dataset
import h5py
import numpy as np 
from PIL import Image
import heatmap_generator

class Dataset(Dataset):
  
  def __init__(self, h5file,model_type,std_filename,mean_filename):
    with h5py.File(h5file, 'r') as hdf:
      # Get the data
      if model_type == "images_landmarks":
        self.input_size = 71
      elif model_type == "landmarks":
        self.input_size = 68
      else:
        self.input_size = 3
      
      
      print("landmarks")
      self.data_lm = np.asarray(hdf.get("landmarks")) if self.input_size != 3 else None
      print("images")
      self.data_im = np.asarray(hdf.get("images")) if self.input_size != 68 else None
      print("expressions")
      self.exprs = np.asarray(hdf.get("expressions"))
    hdf.close()

    # Load mean and std image
    self.mean_image = np.asarray(Image.open(mean_filename)).transpose(2, 0, 1)
    self.std_image = np.asarray(Image.open(std_filename)).transpose(2, 0, 1)

  def getInputSize(self):
    return self.input_size

  def __len__(self):
    return len(self.exprs)


  def __getitem__(self, index):
    # Select sample
    if self.input_size == 3:
      return ((np.asarray(self.data_im[index]) - self.mean_image) / self.std_image),self.exprs[index]
    elif self.input_size == 68:
      return np.asarray(heatmap_generator.getHeatMap(self.data_lm[index])), self.exprs[index]
    else:
       return np.vstack([((np.asarray(self.data_im[index]) - self.mean_image) / self.std_image),np.asarray(heatmap_generator.getHeatMap(self.data_lm[index]))]),self.exprs[index]
    


