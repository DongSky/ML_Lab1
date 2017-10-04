import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import math
#some essential parameters
eps=1e-4
alpha=3e-2
train_size=0
regulaztion_num=1
panel=5e-2
#define the loss function
def Eriri(y,calc_y):
    loss=0
    for i in range(0,len(y)):
        #print("%f  %f"%(y[i],calc_y[i]))
        loss+=(y[i]-calc_y[i])**2
    return 0.5*loss/len(calc_y)
def Eriri_with_panelty(y,calc_y,theta):
    loss=0
    theta_=np.array(theta).reshape(1,order+1)[0]
    for i in range(0,len(y)):
        loss+=(y[i]-calc_y[i])**2
    loss=0.5*loss/len(calc_y)
    for i in range(0,len(theta_)):
        loss+=panel*theta_[i]*theta_[i]
    return loss
#define the CG process
def cg(y,matA):
    theta=np.array([0 for i in y]).reshape(len(y),1)
    y=np.mat(y.reshape(y.shape[0],1))
    k=0
    temp=np.dot(matA,theta)
    r0=y-temp
    eriri=0
    while True:
        k+=1
        if k==1:
            p1=r0
            alpha=float(np.dot(r0.T,r0)/np.dot(np.dot(p1.T,matA),p1))
            theta=theta+alpha*p1
            r1=r0-alpha*np.dot(matA,p1)
            r0=r1
            r=r1
        else:
            p1=r1+float(np.dot(r1.T,r1)/np.dot(r0.T,r0)) * p1
            alpha=float(np.dot(r1.T,r1)/np.dot(np.dot(p1.T,matA),p1))
            theta=theta+alpha*p1
            r2=r1-alpha*np.dot(matA,p1)
            r0=r1
            r1=r2
            r=r2
        if(float(r.T*r)<eps):
            break
        print(r0)
    return theta
if __name__=="__main__":
    #generate argument list, users can set the order of the polynomial before the program starts, default is 9
    parser=argparse.ArgumentParser(description="Linear Curve Fit")
    parser.add_argument('-o',action="store",dest="order",default=9,type=int)
    result=parser.parse_args()
    order=result.order
    #initialize matplotlib to display the curve
    plt.figure()
    xa=[]
    ya=[]
    #read the training data from file
    fi=open("train.txt","r")
    lines=fi.readlines()
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
    xa=np.array(xa)/regulaztion_num
    Mat=[]
    #calculate X matrix
    for i in range(0,2*order+1):
        s=0
        for j in range(0,len(xa)):
            s+=(xa[j]**i)
        Mat.append(s)
    #calculate X.T * X matrix(named as MatA)
    MatA=[]
    for i in range(0,order+1):
        row=Mat[i:i+order+1]
        MatA.append(row)
    MatA=np.array(MatA)
    MatB=[]
    #calculate X.T * Y vector(named as MatB)
    for i in range(0,order+1):
        ty=0.0
        for j in range(0,len(xa)):
            ty+=ya[j]*(xa[j]**i)
        MatB.append(ty)
    MatB=np.array(MatB)

    print(MatA)
    #calculate w with cg
    theta_new=cg(MatB,MatA)
    print(theta_new.reshape(1,order+1)[0])
    #define f(x) to calculate output
    def f(x):
        y=0.0
        for i in range(0,order+1):
            y+=theta_new[i]*(x**i)
        return y
    #draw the curve in matplotlib using a lot of points
    xxa=np.arange(-1.9,1.9,0.01)
    #initialize the x data into [-1,1] to get the correct output
    x_new=xxa/regulaztion_num
    yya=[]
    for i in range(0,len(xxa)):
        yya.append(float(f(x_new[i])))
    yya=np.array(yya)

    x_test=[]
    y_test=[]
    y_test_linear=[]
    #read the test dataset
    fi=open("test.txt","r")
    test_lines=fi.readlines()
    test_size=len(test_lines)
    for line in test_lines:
        piece=line.split(",")
        x_test.append(float(piece[0]))
        y_test.append(float(piece[1]))
    #initialize the x data into [-1,1] to get the correct output
    x_test=np.array(x_test)/regulaztion_num
    for i in range(0,len(x_test)):
        y_test_linear.append(float(f(x_test[i])))
    #calculate the loss in test data of cg
    eriri_test_linear=Eriri(y_test,y_test_linear)
    print("loss:"+str(eriri_test_linear))
    #show the curve
    plt.plot(xxa,yya,color='g',linestyle='-',marker="")
    plt.show()
