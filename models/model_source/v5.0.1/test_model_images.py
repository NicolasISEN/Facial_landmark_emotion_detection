import torch
from cnn import Net32
import h5py
import numpy as np
from torchvision import datasets
from dataset import Dataset
import matrice_de_confusion as mc

input_size = 3
PATH = "./save_model/best_model_images.pth"
h5file = "./test_dataset_pack.h5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net32(input_size).cuda()
model.load_state_dict(torch.load(PATH)["state_dict"])
model.eval()

#data_im = None
#data_lm = None
#exprs = None
#with h5py.File(h5file, 'r') as hdf:
#    print("landmarks")
#    data_lm = np.asarray(hdf.get("landmarks"))
#    print("images")
#    data_im = np.asarray(hdf.get("images"))
#    print("expressions")
#    exprs = np.asarray(hdf.get("expressions"))
#    hdf.close()

test_loader = torch.utils.data.DataLoader(dataset=Dataset('test_dataset_pack.h5',"images","std_training.png","mean_training.png"), batch_size=32, num_workers=0)

y_test = []
y_pred = []

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        #Get the inputs
        images, expressions = data
        y_test += expressions.cpu().numpy().tolist()
        images = images.to(device)
        expressions = expressions.to(device).long()

        #Get the outputs
        output = model(images)
        _,predicted = torch.max(output,1)
        y_pred += predicted.cpu().numpy().tolist()

print(len(y_test))
print(len(y_pred))

class_names = ["Neutral","Happiness","Sadness","Surprise","Fear","Disgust","Anger"]
mc.getMatrix(y_test,y_pred,class_names,plot=True,normalize=False)
