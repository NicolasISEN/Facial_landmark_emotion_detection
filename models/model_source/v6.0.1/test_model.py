import torch
from autoencodeur import AutoEncoder
import h5py
import numpy as np
from torchvision import datasets
from dataset import Dataset
import matrice_de_confusion as mc
import matplotlib.pyplot as plt
from matplotlib import pyplot

input_size = 3
PATH = "save_model/best_model_validation.pth"
h5file = "test_dataset_pack0.h5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder().cuda()
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

test_loader = torch.utils.data.DataLoader(dataset=Dataset('test_dataset_pack0.h5',"std_training.png","mean_training.png"), batch_size=32, num_workers=0)

y_test = []
y_pred = []

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        #Get the inputs
        #print(data)
        images,_, expressions = data
        #print(images.cpu().numpy().shape)
        #print(expressions)
        
        y_test += expressions.cpu().numpy().tolist()
        images = images.to(device)
        expressions = expressions.to(device).long()

        #Get the outputs
        outputs = model(images)
        it = 15
        if i == 12:
            print(outputs[0][it])
            x = (outputs[1][it].cpu().numpy()).tolist()[::2]
            y = outputs[1][it].cpu().numpy().tolist()[1::2]
            plt.imshow(images[it][0],cmap="gray")
            plt.scatter(y,x)
            plt.show()

        _,predicted = torch.max(outputs[0],1)
        y_pred += predicted.cpu().numpy().tolist()

print(len(y_test))
print(len(y_pred))

class_names = ["Neutral","Happiness","Sadness","Surprise","Anger"]
mc.getMatrix(y_test,y_pred,class_names,plot=True,normalize=False)
