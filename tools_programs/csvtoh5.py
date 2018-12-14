import menpo.io as mio
import menpo.shape as msh
import numpy as np
import menpo.feature as mfeat
import h5py
from menpo.image import Image
import tqdm
import random
import matplotlib.pyplot as plt
from threading import Thread

flip_id = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]
debug = False

def flip_landmarks(landmarks):
    global flip_id
    flipped_landmarks = []
    landmarks = landmarks.reshape((68, 2))
    for f in flip_id:
        flipped_landmarks.append(landmarks[f])
    return np.asarray(flipped_landmarks, dtype=np.float32).flatten()


def rotation(image,angle):
    image = image.rotate_ccw_about_centre(angle)
    image = image.resize((128,128))
    return image


def mirrorImage(image):
    global debug
    image = image.mirror()
    if debug:
        # miroir de l'image et des landmarks
        for key in image.landmarks.keys():
            print(key)
    landmarks_mirror = image.landmarks['PTS'].lms.points.flatten()
    # rectification de la numerotion des landmarks
    landmarks_mirror=flip_landmarks(landmarks_mirror)
    image.landmarks['PTS'] = msh.PointCloud(landmarks_mirror.reshape((68, 2)))
    if debug:
        for value in image.landmarks.values():
            print(value)
        for key in image.landmarks.keys():
            print(key)
    return image

def getImage(path,dataLM):
    # charger l'image
    image = mio.import_image(path)
    for i in range(0,len(dataLM),2):
        tmp = dataLM[i]
        dataLM[i]=dataLM[i+1]
        dataLM[i+1]=tmp
    image.landmarks['PTS'] = msh.PointCloud(np.array(dataLM).reshape((68,2)))
    image = image.resize((128,128))
    return image


def modifImage(filename:str,dirpath:str,_from_to=(-1,-1)):
    global debug
    csv_file = [line.rstrip('\r\n') for line in open(filename, 'rU')]
    images=[]
    landmarks=[]
    affects=[]
    expression=[]
    first=0
    last=len(csv_file)
    if _from_to != (-1,-1):
        first = _from_to[0]
        last = _from_to[1]
    print("***",first,last,' out of ',len(csv_file))
    print(csv_file[first])
    pbar = tqdm.tqdm(total=last-first)
    for i in range(first,last):
        if (i%200==0) and debug:
            print("Etape : ",i,"/",len(csv_file))
        try:
            data = csv_file[i].split(',')
        except Exception as e:
            continue
        if len(data) < 5:
            pbar.update(1)
            continue
        if int(data[-4]) == 4 or int(data[-4]) == 5:
            pbar.update(1)
            continue
        im=getImage(dirpath+data[0], [float(i) for i in data[5].split(';') ]) #ATTENTION Pour les landmarks x et y sont inversés
        
        if (float(data[-1])==1):
            im=mirrorImage(im)
        elif (float(data[-1])%2==0 and float(data[-1]) != 0):
            im=rotation(im,random.randint(-25,25))
        elif (float(data[-1])%2==1):
            im=mirrorImage(im)
            im=rotation(im,random.randint(-25,25))
        img = np.array(im.pixels)
        img = np.asarray([0.21*img[0,:,:]+0.72*img[1,:,:]+0.07*img[2,:,:]])
        images.append(img)
        landmarks.append(im.landmarks['PTS'].lms.points)
        expression.append(float(data[-4]))
        affects.append([float(data[-3]),float(data[-2])])
        if debug:
            ppp = []
            line=[]
            for i in range(img.shape[2]):
                for j in range(img.shape[1]):
                    line.append(img[:,i,j])
                ppp.append(np.array(line).tolist())
                line=[]
            ppp = np.asarray(ppp)
            print(ppp.shape)
            plt.imshow(ppp[:,:,0],cmap='gray')
            plt.show()
            #im = im.rasterize_landmarks(marker_style='o', line_width=10)
            #mio.export_image(im,'test'+str(i)+'2.png')
        pbar.update(1)
    pbar.close()
    return images,landmarks,expression,affects

def createh5(dirpath,csvfilename,destinationfilename,nbofdatasets=1):
    global debug
    csv_file = [line.rstrip('\r\n') for line in open(csvfilename, 'rU')]
    size = len(csv_file)
    del csv_file
    _from_to = []
    if nbofdatasets == 1:
        _from_to.append((-1,-1))
    else:
        tmpsize=int(size/nbofdatasets)
        count=0
        for i in range(nbofdatasets):
            _from_to.append((count,count+tmpsize))
            count = (count+tmpsize+1) if (count+tmpsize+1)<size else size
    for i in range(nbofdatasets):
        t = DataSetCreator(csvfilename,destinationfilename+str(i),dirpath,_from_to[i])
        t.run()

class DataSetCreator(object):
    """Thread chargé simplement d'afficher une lettre dans la console."""
    def __init__(self,csvfilename,destinationfilename,dirpath,_from_to):
        self.csv = csvfilename
        self.dest=destinationfilename
        self.dir=dirpath
        self.ft=_from_to

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        global debug
        print(self.ft)
        images,landmarks,expression,affects = modifImage(self.csv,self.dir,_from_to=self.ft)
        if debug:
            print(affects)
            print(landmarks)
            print(images)
            print(expression)
        with h5py.File(self.dest+'.h5', 'w') as T:
            T.create_dataset('images', data=images, dtype=np.float32)
            T.create_dataset('landmarks', data=landmarks, dtype=np.float32)
            T.create_dataset('expressions', data=expression, dtype=np.float32)
            T.create_dataset('affects', data=affects, dtype=np.float32)
            T.close()
        del images,landmarks,expression,affects

    

if __name__ == "__main__":

    dirpath="/home/isen/Documents/PROJET_M1_DL/facial/Affect-Net/MAN/Manually_Annotated_Images/"
    csv_training_dataset = "training_dataset.csv"
    csv_validation_dataset = "validation_dataset.csv"
    csv_test_dataset = "test_dataset.csv"
    destination_training_dataset = "training_dataset_pack"
    destination_validation_dataset = "validation_dataset_pack"
    destination_test_dataset = "test_dataset_pack"
    if not debug:
        createh5(dirpath,csv_training_dataset,destination_training_dataset,4)
        createh5(dirpath,csv_validation_dataset,destination_validation_dataset)
        createh5(dirpath,csv_test_dataset,destination_test_dataset)
    else:
        images,landmarks,expression,affects = modifImage(csv_training_dataset,dirpath)