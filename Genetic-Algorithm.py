# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:12:05 2018

@author: AkashSrivastava
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:13:52 2018

@author: AkashSrivastava
"""
#importing all the packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#method to generate random weights
def weight_generation(size,n):
    weight=[]
    for i in range(0,size):
        weight.append(np.random.uniform(-1,1,(5,n)))
    return weight    
  
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
#Method to calculate fitness value
def cal_fitness_value(X_training,weight,Y_training,n):    
    prod=[]
    for i in range(0,len(weight)):
        prod.append(np.dot(X_training,weight[i]))
    np.shape(prod)

    Y_fitAll=[]
    for x in range(0,len(weight)):
        Y_hat=[]
        for row in range(0,len(X_training)):
            fn=0
            for col in range(0,n):
                fn=fn+(1/(1+math.exp(-prod[x][row][col])))
            Y_hat.append(fn) 
        y=0
        fit=0
        y1=0
        for rows in range(0,len(Y_training)):
            x=np.array(Y_training)
            y1=Y_hat[rows]-x[rows]
            y=y+math.pow(y1,2)
            x=y/189
            fit=(1-x)*100
        Y_fitAll.append(fit)
    return Y_fitAll







#method to normalize the weights
def decimal_to_Normal(weight):
    weight_normal=[]
    for i in range(0,len(weight)):
        weight_nor=normalize(weight[i])
        ####### for values between 0 and 1
        weight_nor=(weight_nor+2)/4

        ################# multiplying by 1000

        weight_nor=np.round(weight_nor*1000).astype(int)
        weight_normal.append(weight_nor)
    return weight_normal

#method to convert normalized weights to binary for construction of chromosomes
def to_binary_chromosome(weight_normal):
    chromosomes=[]
    for x in range(0,len(weight_normal)):
       # weight_bin=np.empty((5,10),dtype='|S10')
        str2=""
        for row in range(0,5):
            for col in range(0,10):
                str1=(str(bin(weight_normal[x][row][col])[2:].zfill(10)))
               # weight_bin[row,col]=str1
                str2=str2+str1 
        chromosomes.append(str2)
    return chromosomes







'''
a='123456'
b='78922456'

np.hstack((a[:3],b[3:]))[0]+np.hstack((a[3:],b[:3]))[1]
'''



#method to perform crossover operation
def crossover(chromosomes,parent):
    child=[]
    for i in range(0,len(chromosomes)):
        c_point=np.random.random_integers(2,len(sire_chromosome)-1)
    
        child.append(np.hstack((parent[:c_point],chromosomes[i][c_point:]))[0]+np.hstack((parent[:c_point],chromosomes[i][c_point:]))[1])
        child.append(np.hstack((parent[c_point:],chromosomes[i][:c_point]))[0]+np.hstack((parent[c_point:],chromosomes[i][:c_point]))[1] )
    return child
'''


child1=list(child[0])
child1[0]=child1[0].replace(child1[0],'1')
child[0]="".join(child1)
'''

#method to perpform mutation
#for mutation taking 5% of 500 bits that is 25 bits
def mutate(child):
    for i in range(0,len(child)):
        for x in range(0,25):
            bit=np.random.random_integers(x,len(sire_chromosome)-1)
            if bit==x:
                child_chromosome=list(child[i])
                child_chromosome[bit]=child_chromosome[bit].replace(child_chromosome[bit],str(1-int(child_chromosome[bit])))
                child[i]="".join(child_chromosome)
    return child

#method for desegmentation of chromosomes and decimal to binary and dividing them by 1000 and denormalization
def binary_to_decimal(child):
    child_desegmented=[]
    for i in range(0,len(child)):
        ind=10
        child_deseg=[]
        for x in range(0,5):
            for j in range(0,10):
                child_deseg.append(((int(child[i][ind-10:ind],2)/1000)*4)-2)#here converting to -2 to 2 instead of -1 to 1 as given in discussion(on blackboard) to improve fitness value
                ind=ind+10        
        child_desegmented.append(np.array(child_deseg).reshape(5,10))
    return child_desegmented


#### selecting top 500 offspring based on fitness value
def top_child(fitness_child,child):
    Y_fitAll_child_child_desegmented=pd.Series(fitness_child)
   
    Y_fitAll_child_child_desegmented=Y_fitAll_child_child_desegmented.sort_values(ascending=False)[:500]

    child_to_parent=[]
    for x in Y_fitAll_child_child_desegmented.index:
        child_to_parent.append(child[x])
    return child_to_parent


#---------------------------------program----------------------------------
#Step 1
#importing data
data=pd.read_csv("E:/ucd/python/project/project1dataset.csv")
data

#step 2
#selecting first 5 columns and last are target
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

#step 3
N=10


#step 4
#dividing the training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df,target,test_size=0.25)

#step 5
#normalizing the training and testing data
from sklearn.preprocessing import normalize
X_train_norm=normalize(X_train)
Y_train_norm=normalize(Y_train)
Y_train_norm=np.transpose(Y_train_norm)
X_test_norm=normalize(X_test)
Y_test_norm=normalize(Y_test)
Y_test_norm=np.transpose(Y_test_norm)

#np.shape(Y_train)
#X_train_norm=X_train_norm*10
#np.shape(X_train_norm)

#step 6 and 7
#defining Npop
Npop=500

#weight generation
weight=weight_generation(Npop,N)
np.shape(weight)

#step 8
# fitness value calculation
first_fitness_value=cal_fitness_value(X_train_norm,weight,Y_train_norm,N)
np.shape(first_fitness_value)

#Step 9
#Selecting Parent
sire=max(first_fitness_value)
sire_index=first_fitness_value.index(sire)

#Step 10 And 11
#Binarizing the population
weight_normal=decimal_to_Normal(weight)
np.shape(weight_normal)

#Generating chromosomes of binary after binarizing the population 
chromosomes=to_binary_chromosome(weight_normal)
np.shape(chromosomes)




#sire_chromosome=chromosomes[sire_index]
#len(chromosomes[156])
parent=chromosomes[sire_index]


#Step 18 iteration
i=0
ite=[]
f_val=[]
while(i<50):#using 50 iterations as fitness value plateaus around 30 iterations
    #step 12
    #performing crossover
    #child=chromosomes
    #child=child+crossover(chromosomes,parent)
    child=crossover(chromosomes,parent)
    len(child)


    #step 13
    #performing mutation
    child=mutate(child)
    len(child)

    #step 14
    #performing debinarization of the offsprings population
    child_desegmented=binary_to_decimal(child)
    np.shape(child_desegmented)
    
    
    #Step 15
    #calculating fitness value of population
    fitness_child=cal_fitness_value(X_train_norm,child_desegmented,Y_train_norm,N)
    np.shape(fitness_child)


    #Step 16
    #selecting top 500 population According to fitness value
    sire1=max(fitness_child)
    sire_index1=fitness_child.index(sire1)    
    child_to_parent=top_child(fitness_child,child)
    np.shape(child_to_parent)

    
    #Step 17
    #Selecting parent   
    chromosomes=child_to_parent
    if sire1>sire:
        sire=sire1
        sire_index=sire_index1
    parent=child[sire_index]
    print("Parent fitness value for iteration ",i," : ",sire)
    ite.append(i)
    f_val.append(sire)
    
    
    #updating i value
    i=i+1


#plotting scatterplot
plt.scatter(ite, f_val, alpha=0.5)
plt.show()    

#calculating y_hat for testing data
prod_test=[]
for i in range(0,len(child_to_parent)):
    prod_test.append(np.dot(X_test_norm,child_desegmented[sire_index]))
    
    np.shape(prod_test)


for x in range(0,len(child_to_parent)):
    
    Y_hat_test=[]
    for row in range(0,len(X_test_norm)):
        fn=0
        for col in range(0,10):
            fn=fn+(1/(1+math.exp(-prod_test[x][row][col])))
        Y_hat_test.append(fn)
        
np.shape(Y_hat_test)
#Y_hat_test=np.transpose(normalize(Y_hat_test))#### have doubt here
# 3-d scatter plot
fig=plt.figure()
ax=plt.axes( projection='3d')
ax.scatter3D(X_test_norm[:,0],X_test_norm[:,1],Y_hat_test)
ax.scatter3D(X_test_norm[:,0],X_test_norm[:,1],Y_test_norm)
plt.show()



# calculating error on test data
s1=0
for i in range(0,len(Y_hat_test)):
    s=Y_hat_test[i]-Y_test_norm[i]
    s1=s1+math.pow(s,2)

error=s1/len(Y_test_norm)
print("overall error:",error)
