# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:13:52 2018

@author: AkashSrivastava
"""
import pandas as pd
import numpy as np
import math


data=pd.read_csv("project1dataset.csv")
data
weight=data["Weight lbs"]
height=data["Height inch"]
neck_cir=data["Neck circumference"]
chest=data["Chest circumference"]
abdomen=data["Abdomen  circumference"]
hip=data["Hip circumference"]
target=data["bodyfat"]
type(target)
df=pd.DataFrame(data=[weight,height,neck_cir,chest,abdomen])
df=pd.DataFrame.transpose(df)



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df,target,test_size=0.25)


from sklearn.preprocessing import normalize
X_train_norm=normalize(X_train)
Y_train_norm=normalize(Y_train)
Y_train_norm=np.transpose(Y_train_norm)
np.shape(Y_train)

#X_train_norm=X_train_norm*10
np.shape(X_train_norm)


################################################################################



################################################################################
weight=[]
for i in range(0,500):
    weight.append(np.random.uniform(-1,1,(5,10)))

np.shape(weight)
'''
Npop_normal1=normalize(Npop1)
Npop_normal1=(Npop_normal1+1)/2
#Npop_normal1=np.round(Npop_normal1*1000).astype(int)

np.shape(Npop_normal1)

Npop_bin=np.array(np.ones(50))

for row in range(0,10):
    for col in range(0,5):
        x123=bin(Npop_normal1[row,col])[2:].zfill(10)
        Npop_normal1[row][col]=x123
       #print(bin(Npop_normal[row][col])[2:].zfill(10))
   '''     
#################################################################################


Npop=500

############################
prod=[]
for i in range(0,500):
    prod.append(np.dot(X_train_norm,weight[i]))
np.shape(prod)

Y_fitAll=[]
for x in range(0,500):
    Y_hat=[]
    for row in range(0,189):
        fn=0
        for col in range(0,10):
            fn=fn+(1/(1+math.exp(-prod[x][row][col])))
        Y_hat.append(fn) 
    y=0
    fit=0
    y1=0
    for rows in range(0,189):
        x=np.array(Y_train_norm)
        y1=Y_hat[rows]-x[rows]
        y=y+math.pow(y1,2)
        x=y/189
        fit=(1-x)*100
    Y_fitAll.append(fit)
            

Y_train_norm
np.shape(Y_hat)


##############################################################################step 9
sire=max(Y_fitAll)
sire_index=Y_fitAll.index(sire)
#######################################################################step 10
weight_normal=[]
for i in range(0,500):
    weight_nor=normalize(weight[i])
################### for values between 0 and 1
    weight_nor=(weight_nor+1)/2

######################## multiplying by 1000

    weight_nor=np.round(weight_nor*1000).astype(int)
    weight_normal.append(weight_nor)
############################################################## to binary and chromosome
np.shape(weight_normal)
chromosomes=[]
for x in range(0,500):
    weight_bin=np.empty((5,10),dtype='|S10')
    str2=""
    for row in range(0,5):
        for col in range(0,10):
            str1=(str(bin(weight_normal[x][row][col])[2:].zfill(10)))
            weight_bin[row,col]=str1
            str2=str2+str1 
    chromosomes.append(str2)

np.shape(chromosomes)
sire_chromosome=chromosomes[sire_index]
len(chromosomes[156])



###########################################################################################################crossover
'''
a='123456'
b='78922456'

np.hstack((a[:3],b[3:]))[0]+np.hstack((a[:3],b[3:]))[1]
'''





child=[]
for i in range(0,500):
    c_point=np.random.random_integers(2,len(sire_chromosome)-1)
    
    child.append(np.hstack((chromosomes[sire_index][:c_point],chromosomes[i][c_point:]))[0]+np.hstack((chromosomes[sire_index][:c_point],chromosomes[i][c_point:]))[1])
    child.append(np.hstack((chromosomes[sire_index][c_point:],chromosomes[i][:c_point]))[0]+np.hstack((chromosomes[sire_index][c_point:],chromosomes[i][:c_point]))[1])

'''
len(child)

child1=list(child[0])
child1[0]=child1[0].replace(child1[0],'1')
child[0]="".join(child1)
'''
#########################################################################################################mutation
#for mutation taking 10% of 500 bits that is50 bits
for i in range(0,1000):
    for x in range(0,50):
        bit=np.random.random_integers(x,len(sire_chromosome)-1)
        if bit==x:
            child_chromosome=list(child[0])
            child_chromosome[bit]=child_chromosome[bit].replace(child_chromosome[bit],str(1-int(child_chromosome[bit])))
            child[0]="".join(child_chromosome)


########################## De degmentation of chromosomes and decimal to binary and dividing them by 1000 and de normalization


child_desegmented=[]
for i in range(0,1000):
    ind=10
    child_deseg=[]
    for x in range(0,5):
        for j in range(0,10):
            child_deseg.append(((int(child[i][ind-10:ind],2)/1000)*2)-1)
            ind=ind+10        
    child_desegmented.append(np.array(child_deseg).reshape(5,10))

np.shape(child_desegmented[0])
max()

###########################fitness value
prod_child=[]
for i in range(0,1000):
    prod_child.append(np.dot(X_train_norm,child_desegmented[i]))
np.shape(prod_child)

Y_fitAll_child=[]
for x in range(0,1000):
    Y_hat_child=[]
    for row in range(0,189):
        fn=0
        for col in range(0,10):
            fn=fn+(1/(1+math.exp(-prod_child[x][row][col])))
        Y_hat_child.append(fn) 
    y=0
    fit=0
    y1=0
    for rows in range(0,189):
        x=np.array(Y_train_norm)
        y1=Y_hat_child[rows]-x[rows]
        y=y+math.pow(y1,2)
        x=y/189
        fit=(1-x)*100
    Y_fitAll_child.append(fit)
            

np.shape(Y_fitAll_child[0])
####################################################### sorting according to fitness value and removing chromosomes

Y_fitAll_child_child_desegmented=pd.Series(Y_fitAll_child)
max(Y_fitAll_child_child_desegmented)
Y_fitAll_child_child_desegmented=Y_fitAll_child_child_desegmented.sort_values(ascending=False)[:500]

child_to_parent=[]
for x in Y_fitAll_child_child_desegmented.index:
    child_to_parent.append(child_desegmented[x])

sire1=max(Y_fitAll_child)
sire_index1=Y_fitAll_child.index(sire1)
np.shape(child_to_parent)



##################################################################################################
if sire1>sire:
    parent_chromosome=child[sire_index1]






