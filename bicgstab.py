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

def bicgstab_penalty(y,calc_y,matX,theta):
    y=np.mat(y.reshape(y.shape[0],1))
    y_new=[]
    for i in calc_y:
        y_new.append(i)
    y_new=np.array(y_new)
    y_new=np.mat(y_new.reshape(y_new.shape[0],1))
    eriri=Eriri_with_panelty(y,y_new,theta)
    print(y)
    print(y_new)
    r0=y-y_new
    r0__=y-y_new
    rho0=1
    alpha0=1
    w0=1
    v0=np.zeros(len(y)).reshape(len(y),1)
    #print(r0.shape)
    p0=np.zeros(len(y)).reshape(len(y),1)
    while True:
        #print(np.dot(r0__.T,r0))
        rho1=np.array(np.dot(r0__.T,r0))[0][0]
        #print(rho1)
        beta=(rho1/rho0)*(alpha0/w0)
        #print(r0)
        #print(p0.shape)
        p1=r0+beta*(p0-w0*v0)
        #print(p1)
        #print(theta.shape)
        v1=np.dot(matX,p1)
        temp=float((np.dot(r0__.T,v1))[0][0])
        #print(temp)
        alpha0=rho1/temp
        #print(alpha0)
        h=theta+alpha0*p1
        #print(h)
        y_new=np.array(np.dot(matX,np.array(h).reshape(len(h),1)).reshape(1,len(y)))[0]
        new_eriri=Eriri_with_panelty(y,y_new,h)
        if abs(new_eriri-eriri)<eps:
            theta=h
            break;
        s=r0-alpha0*v1
        t=np.dot(matX,s)
        w1=np.array(np.dot(t.T,s))[0][0]/np.array(np.dot(t.T,t))[0][0]
        theta=h+w1*s
        y_new=np.array(np.dot(matX,np.array(theta).reshape(len(theta),1)).reshape(len(y),1))
        if abs(new_eriri-eriri)<eps:
            break
        r0=s-w1*t
        rho0=rho1
        p0=p1
        v0=v1
        w0=w1
        eriri=new_eriri
        print(eriri)
    return np.dot(matX,theta),theta
def bicgstab(y,calc_y,matX,theta):
    pass
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
    xa=np.array(xa)/regulaztion_num
    Mat=[]
    for i in range(0,2*order+1):
        s=0
        for j in range(0,len(xa)):
            s+=(xa[j]**i)
        Mat.append(s)
    MatA=[]
    for i in range(0,order+1):
        row=Mat[i:i+order+1]
        MatA.append(row)
    MatA=np.array(MatA)
    print(xa)
    print(MatA)
    MatB=[]
    for i in range(0,order+1):
        ty=0.0
        for j in range(0,len(xa)):
            ty+=ya[j]*(xa[j]**i)
        MatB.append(ty)
    MatB=np.array(MatB)
    theta=[]
    for i in range(0,10):
        theta.append(random.uniform(-1,1))
    theta0=np.mat(np.array(theta).reshape(order+1,1))
    y_new=np.array(np.dot(MatA,theta0).reshape(1,order+1))[0]
    theta=np.array(theta0.reshape(1,order+1))[0]
    y_bgd,theta_new=bicgstab_penalty(y_new,MatB,MatA,theta0)
    print(theta_new.reshape(1,order+1)[0])
