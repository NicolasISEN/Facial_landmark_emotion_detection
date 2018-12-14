import torch
from torch.utils.data import Dataset
import h5py
import numpy as np 
from PIL import Image
import heatmap_generator

class Dataset(Dataset):
  
  def __init__(self, h5file,model_type):
    with h5py.File(h5file, 'r') as hdf:

      #Setting training data and validation data
      #length = len(hdf.get("expressions"))
      #shuff = np.arange(length)
      #np.random.shuffle(shuff)
      shuff = np.load('shuff.npy')
      threshold = int(len(shuff)*0.8)

      # Get the data
      if model_type == "images_landmarks":
        self.input_size = 71
      elif model_type == "landmarks":
        self.input_size = 68
      else:
        self.input_size = 3
      
      
      print("landmarks")
      self.training_data_lm = np.asarray(hdf.get("landmarks"))[shuff[:threshold]] if self.input_size != 3 else None
      self.validation_data_lm = np.asarray(hdf.get("landmarks"))[shuff[threshold:]] if self.input_size != 3 else None
      #print("landmarks_inverse")
      #if self.input_size != 3:
      #  self.training_data_lm[:,[0,1]] = self.training_data_lm[:,[1,0]]
      #  self.validation_data_lm[:,[0,1]] = self.validation_data_lm[:,[1,0]]
      print("images")
      self.training_data_im = np.asarray(hdf.get("images"))[shuff[:threshold]] if self.input_size != 68 else None
      self.validation_data_im = np.asarray(hdf.get("images"))[shuff[threshold:]] if self.input_size != 68 else None
      print("expressions")
      self.training_exprs = np.asarray(hdf.get("expressions"))[shuff[:threshold]]
      self.validation_exprs = np.asarray(hdf.get("expressions"))[shuff[threshold:]]
    hdf.close()
    self.isTraining = True

    # Load mean and std image
    self.mean_image = np.asarray(Image.open("mean_image.png")).transpose(2, 0, 1)
    self.std_image = np.asarray(Image.open("std_image.png")).transpose(2, 0, 1)

  def getInputSize(self):
    return self.input_size

  def shuffle_training(self):
    shuff = np.arange(len(self.training_exprs))
    np.random.shuffle(shuff)
    self.training_exprs = self.training_exprs[shuff]
    if self.input_size != 68:
      self.training_data_im = self.training_data_im[shuff]
    if self.input_size != 3:
      self.training_data_lm = self.training_data_lm[shuff]

  def __len__(self):
    return (len(self.training_exprs) if self.isTraining else len(self.validation_exprs))

  def training(self):
    self.isTraining = True

  def validation(self):
    self.isTraining = False

  def __getitem__(self, index):
    # Select sample
    if self.isTraining:
      if self.input_size == 3:
        return ((np.asarray(self.training_data_im[index]) - self.mean_image) / self.std_image),self.training_exprs[index]
      elif self.input_size == 68:
        return np.asarray(heatmap_generator.getHeatMap(self.training_data_lm[index])), self.training_exprs[index]
      else:
        return np.vstack([((np.asarray(self.training_data_im[index]) - self.mean_image) / self.std_image),np.asarray(heatmap_generator.getHeatMap(self.training_data_lm[index]))]),self.training_exprs[index]
    else:
      if self.input_size == 3:
        return ((np.asarray(self.validation_data_im[index]) - self.mean_image) / self.std_image),self.validation_exprs[index]
      elif self.input_size == 68:
        return np.asarray(heatmap_generator.getHeatMap(self.validation_data_lm[index])), self.validation_exprs[index]
      else:
        return np.vstack([((np.asarray(self.validation_data_im[index]) - self.mean_image) / self.std_image),np.asarray(heatmap_generator.getHeatMap(self.validation_data_lm[index]))]),self.validation_exprs[index]


