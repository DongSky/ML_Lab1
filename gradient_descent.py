import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import math

eps=1e-8
alpha=3e-2
train_size=0
regulaztion_num=1
panel=5e-2

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
    for i in range(0,len(theta_)):
        loss+=panel*theta_[i]/(regulaztion_num**i)*theta_[i]/(regulaztion_num**i)
    return 0.5*loss/len(calc_y)
def transform(B):
    theta_=(np.array(B).reshape(1,order+1))[0]
    b=[]
    for i in range(order+1):
        b.append(theta_[i]/(2.5**i))
    return np.mat(np.array(b).reshape(order+1,1))
def BGD_penalty(y,calc_y,matX,theta):
    print(regulaztion_num)
    y_new=[]
    for i in calc_y:
        y_new.append(i)
    eriri=Eriri_with_panelty(y,y_new,theta)
    X=np.array(matX)
    cnt=1
    while True:
        new_theta=[]
        for i in range(0,len(theta)):
            s=0.0
            if i==0:
                for j in range(0,len(y)):
                    s+=((y_new[j]-y[j])*X[j][i])
            else:
                for j in range(0,len(y)):
                    s+=((y_new[j]-y[j])*X[j][i])+panel/(train_size*theta[i]*(regulaztion_num**i))
            flag=theta[i]-alpha*s/train_size;
            new_theta.append(flag)
        theta=new_theta
        y_new=np.array(np.dot(matX,np.array(new_theta).reshape(len(theta),1)).reshape(1,len(y)))[0]
        new_eriri=Eriri_with_panelty(y,y_new,theta)
        print("%05d    %.8f"%(cnt,new_eriri))
        cnt+=1
        if abs(new_eriri-eriri)<eps:
            break
        eriri=new_eriri
    return y_new,transform(theta)
def BGD(y,calc_y,matX,theta):
    print(train_size)
    y_new=[]
    for i in calc_y:
        y_new.append(i)
    eriri=Eriri(y,y_new)
    X=np.array(matX)
    cnt=1
    while True:
        new_theta=[]
        for i in range(0,len(theta)):
            s=0.0
            for j in range(0,len(y)):
                s+=((y_new[j]-y[j])*X[j][i])
            flag=theta[i]-alpha*s/train_size;
            new_theta.append(flag)
        theta=new_theta
        y_new=np.array(np.dot(matX,np.array(new_theta).reshape(len(theta),1)).reshape(1,len(y)))[0]
        new_eriri=Eriri(y,y_new)
        print("%05d    %.8f"%(cnt,new_eriri))
        cnt+=1
        if abs(new_eriri-eriri)<eps:
            break
        eriri=new_eriri
    return y_new,transform(theta)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Linear Curve Fit")
    parser.add_argument('-o',action="store",dest="order",default=9,type=int)
    result=parser.parse_args()
    order=result.order
    fig=plt.figure()
    ax=fig.add_subplot(111)
    xa=[]
    ya=[]
    fi=open("train.txt","r")
    lines=fi.readlines()
    train_size=len(lines)
    for line in lines:
        piece=line.split(",")
        xa.append(float(piece[0]))
        ya.append(float(piece[1]))
    ax.plot(xa,ya,color='m',linestyle='',marker='.')
    maxn=0.0
    for i in xa:
        maxn=max(maxn,abs(i))
    regulaztion_num=max(regulaztion_num,math.ceil(maxn))
    x_regu=np.array(xa)/regulaztion_num
    #print(x_regu)
    ya=np.array(ya)
    #print(ya)
    MatX=[]
    for i in range(0,order+1):
        MatX.append(np.array(x_regu)**i)
    MatX=np.mat(MatX).T
    theta=[]
    for i in range(0,10):
        theta.append(random.uniform(-1,1))
    theta0=np.mat(np.array(theta).reshape(order+1,1))
    y_new=np.array(np.dot(MatX,theta0).reshape(1,len(ya)))[0]
    theta=np.array(theta0.reshape(1,order+1))[0]
    y_bgd,theta_new=BGD_penalty(y_new,ya,MatX,theta)
    print(theta_new.reshape(1,order+1)[0])
