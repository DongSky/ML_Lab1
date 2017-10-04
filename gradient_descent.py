import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import math
#some essential parameters
eps=5e-8
alpha=1e-3
train_size=0
regulaztion_num=1
panel=3e-3
#loss function without penalty
def Eriri(y,calc_y):
    loss=0
    for i in range(0,len(y)):
        loss+=(y[i]-calc_y[i])**2
    return 0.5*loss/len(calc_y)
#loss function with penalty
def Eriri_with_panelty(y,calc_y,theta):
    loss=0
    theta_=np.array(theta).reshape(1,order+1)[0]
    for i in range(0,len(y)):
        loss+=(y[i]-calc_y[i])**2
    loss=0.5*loss/len(calc_y)
    for i in range(0,len(theta_)):
        loss+=panel*theta_[i]*theta_[i]
    return loss
#transform the parameter w into another w so that we can calculate f(x) directly without the regulation process
def transform(B):
    theta_=(np.array(B).reshape(1,order+1))[0]
    b=[]
    for i in range(order+1):
        b.append(theta_[i]/(regulaztion_num**i))
    return np.mat(np.array(b).reshape(order+1,1))
#gradient descent with penalty
def BGD_penalty(y,matX,theta):
    print(train_size)
    X=np.array(matX)
    cnt=1
    eriri=0
    y=np.array(y).reshape(len(y),1)
    while True:
        #update w(named as theta)
        theta=theta-alpha*(X.T*(X*theta-y))/train_size
        #calculate the new loss
        new_eriri=Eriri_with_panelty(y,X*theta,theta)
        #output the current loss
        if cnt%1000==0:
            print("%05d    %.8f"%(cnt,new_eriri))
        cnt+=1
        #check the exit condition
        if abs(new_eriri-eriri)<eps:
            break
        #swap the loss value, prepare for next iteration
        eriri=new_eriri
        #return the parameter with transform
    return transform(theta)
def BGD(y,matX,theta):
    print(train_size)
    X=np.array(matX)
    cnt=1
    eriri=0
    y=np.array(y).reshape(len(y),1)
    while True:
        #update w(named as theta)
        theta=theta-alpha*(X.T*(X*theta-y))/train_size
        #calculate the new loss
        new_eriri=Eriri(y,X*theta)
        #output the current loss
        if cnt%1000==0:
            print("%05d    %.8f"%(cnt,new_eriri))
        cnt+=1
        #check the exit condition
        if abs(new_eriri-eriri)<eps:
            break
        #swap the loss value, prepare for next iteration
        eriri=new_eriri
        #return the parameter with transform
    return transform(theta)
if __name__=="__main__":
    #generate argument list, users can set the order of the polynomial before the program starts, default is 9
    parser=argparse.ArgumentParser(description="Linear Curve Fit")
    parser.add_argument('-o',action="store",dest="order",default=9,type=int)
    result=parser.parse_args()
    order=result.order
    #initialize matplotlib to display the curve
    plt.figure("MachineLearningProjectOne")
    xa=[]
    ya=[]
    #read the training data from file
    fi=open("train.txt","r")
    lines=fi.readlines()
    fi.close()
    train_size=len(lines)
    for line in lines:
        piece=line.split(",")
        xa.append(float(piece[0]))
        ya.append(float(piece[1]))
    plt.plot(xa,ya,color='m',linestyle='',marker='.')
    #set the regulaztion_num to change every x into [-1,1]
    maxn=0.0
    for i in xa:
        maxn=max(maxn,abs(i))
    regulaztion_num=max(regulaztion_num,math.ceil(maxn))
    x_regu=np.array(xa)/regulaztion_num
    ya=np.array(ya)
    #calculate the X matrix
    MatX=[]
    for i in range(0,order+1):
        MatX.append(np.array(x_regu)**i)
    MatX=np.mat(MatX).T
    #initialize the w(named theta & theta0) parameter with random value
    theta=[]
    for i in range(0,order+1):
        theta.append(random.uniform(-1,1))
    theta0=np.mat(np.array(theta).reshape(order+1,1))
    #calculate w with penalty
    theta_new=BGD_penalty(ya,MatX,theta0)
    print(theta_new.reshape(1,order+1)[0])
    #define f(x) to calculate output
    def f(x):
        y=0.0
        for i in range(0,order+1):
            y+=theta_new[i]*(x**i)
        return y
    #display the curve with points
    xxa=[]
    poi=-1.9
    while(poi<=1.9):
        xxa.append(poi)
        poi+=0.01
    xxa=np.array(xxa).reshape(len(xxa),1)
    yya=[]
    for i in range(0,len(xxa)):
        yya.append(f(xxa[i]))
    yya=np.array(yya).reshape(len(yya),1)
    plt.plot(xxa,yya,color='g',linestyle='-',marker="")
    #initialize the w(named theta & theta1) parameter with random value
    theta1=np.mat(np.array(theta).reshape(order+1,1))
    #calculate the w with penalty
    theta_new_1=BGD(ya,MatX,theta1)
    print(theta_new_1.reshape(1,order+1)[0])
    #define another f(x) to calculate output which is optimized by gradient penalty
    def futa(x):
        y=0.0
        for i in range(0,order+1):
            y+=theta_new_1[i]*(x**i)
        return y
    #display another curve with points
    xxa1=[]
    poi=-1.9
    while(poi<=1.9):
        xxa1.append(poi)
        poi+=0.01
    xxa1=np.array(xxa1).reshape(len(xxa1),1)
    yya1=[]
    for i in range(0,len(xxa1)):
        yya1.append(futa(xxa1[i]))
    yya1=np.array(yya1).reshape(len(yya1),1)
    print(regulaztion_num)
    plt.plot(xxa1,yya1,color='b',linestyle='-',marker="")

    x_test=[]
    y_test=[]
    #read the test dataset
    fi=open("test.txt","r")
    test_lines=fi.readlines()
    test_size=len(test_lines)
    for line in test_lines:
        piece=line.split(",")
        x_test.append(float(piece[0]))
        y_test.append(float(piece[1]))
    y_test_bgd=[futa(x) for x in x_test]
    y_test_bgd_p=[f(x) for x in x_test]
    #calculate the loss in test data of curve_with_penalty and curve_without_penalty
    eriri_test_bgd=Eriri(y_test,y_test_bgd)
    eriri_test_bgd_p=Eriri(y_test,y_test_bgd_p)
    print("Without Penalty:"+str(eriri_test_bgd))
    print("With Penalty:"+str(eriri_test_bgd_p))
    #show the curve
    plt.legend(loc=4)
    plt.show()
