import numpy as np
import csv
from random import shuffle
import math

csv_reader = None
debug = True
fieldnames = ['filepath', 'facex','facey','facewidth','faceheight','landmarks','expression','valence','arousal','agrandir']

with open('training.csv', mode='r') as csv_file:
    csv_reader = list(csv.DictReader(csv_file,fieldnames=fieldnames))
    csv_file.close()

with open('validation.csv', mode='r') as csv_file:
    csv_reader.append(list(csv.DictReader(csv_file,fieldnames=fieldnames)))
    csv_file.close()


quota = 21000
threshold_test = 800 
threshold_val = 200

csv_reader = csv_reader[:-1]
#Echange des coordonnée x et y
for row in csv_reader:
    tmp = row["landmarks"]
    tmp2 = tmp.split(';')
    #for i in range(0,len(tmp2)-1,2):
    #    tmp2[i],tmp2[i+1]=tmp2[i+1],tmp2[i]
    row["landmarks"] = ';'.join(tmp2)
    row["agrandir"] = 0

array = np.asarray(csv_reader)

exprs = [None]*7
exprs[0] = list(filter(lambda x:int(x["expression"]) == 0,array))[:quota+threshold_test+threshold_val]
exprs[1] = list(filter(lambda x:int(x["expression"]) == 1,array))[:quota+threshold_test+threshold_val]
exprs[2] = list(filter(lambda x:int(x["expression"]) == 2,array))[:quota+threshold_test+threshold_val]
exprs[3] = list(filter(lambda x:int(x["expression"]) == 3,array))[:quota+threshold_test+threshold_val]
exprs[4] = list(filter(lambda x:int(x["expression"]) == 4,array))[:quota+threshold_test+threshold_val]
exprs[5] = list(filter(lambda x:int(x["expression"]) == 5,array))[:quota+threshold_test+threshold_val]
exprs[6] = list(filter(lambda x:int(x["expression"]) == 6,array))[:quota+threshold_test+threshold_val]

expr_train = [None]*7
expr_test = [None]*7
expr_val = [None]*7

expr_test[0] = exprs[0][:threshold_test]
expr_test[1] = exprs[1][:threshold_test]
expr_test[2] = exprs[2][:threshold_test]
expr_test[3] = exprs[3][:threshold_test]
expr_test[4] = exprs[4][:threshold_test]
expr_test[5] = exprs[5][:threshold_test]
expr_test[6] = exprs[6][:threshold_test]

expr_val[0] = exprs[0][threshold_test:threshold_val+threshold_test]
expr_val[1] = exprs[1][threshold_test:threshold_val+threshold_test]
expr_val[2] = exprs[2][threshold_test:threshold_val+threshold_test]
expr_val[3] = exprs[3][threshold_test:threshold_val+threshold_test]
expr_val[4] = exprs[4][threshold_test:threshold_val+threshold_test]
expr_val[5] = exprs[5][threshold_test:threshold_val+threshold_test]
expr_val[6] = exprs[6][threshold_test:threshold_val+threshold_test]

expr_train[0] = exprs[0][threshold_val+threshold_test:]
expr_train[1] = exprs[1][threshold_val+threshold_test:]
expr_train[2] = exprs[2][threshold_val+threshold_test:]
expr_train[3] = exprs[3][threshold_val+threshold_test:]
expr_train[4] = exprs[4][threshold_val+threshold_test:]
expr_train[5] = exprs[5][threshold_val+threshold_test:]
expr_train[6] = exprs[6][threshold_val+threshold_test:]

del exprs

print(np.array(expr_train).shape)
print(np.array(expr_val).shape)
print(np.array(expr_test).shape)

i=1
newtrain = []*7
for expr in expr_train:
    l = len(expr)
    agrandirnfois = int(math.ceil(quota/l))
    print(l,agrandirnfois)
    for j in range(agrandirnfois):
        #print(j)
        for k in range(len(expr)):
            tmp = dict(expr[k])
            tmp["agrandir"]=j
            newtrain.append(tmp)

        #print(expr[0]["agrandir"])
        
        #print("..."+str(newtrain[-1]))
    #print("..."+str(newtrain[-1]))
    newtrain = newtrain[:quota*i]
    i+=1
    #print("***"+str(newtrain[-1]))
print("..."+str(newtrain[87000]))
del expr_train
print("..."+str(newtrain[87000]))
#print(expr_train[5][0])


print(np.array(newtrain).shape)
print(np.array(expr_val).shape)
print(np.array(expr_test).shape)

expr_train = np.asarray(newtrain).flatten()
expr_val = np.asarray(expr_val).flatten()
expr_test = np.asarray(expr_test).flatten()

shuffle(expr_train)
shuffle(expr_val)
shuffle(expr_test)

print(len(expr_train))
print(len(expr_val))
print(len(expr_test))

if debug:
    s = "" 
    s2 = "" 
    count = 0
    print(expr_train[0])
    for i in expr_train:
    #for i in exprs_val:
    #for i in exprs_test:
        count+=1
    print(s,s2,count)

with open("training_dataset.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    #writer.writeheader()
    writer.writerows(expr_train)
    f.close()

with open("validation_dataset.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    #writer.writeheader()
    writer.writerows(expr_val)
    f.close()

with open("test_dataset.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    #writer.writeheader()
    writer.writerows(expr_test)
    f.close()