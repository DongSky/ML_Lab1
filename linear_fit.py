import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

eps=1e-8
alpha=3e-2
train_size=0
regulaztion_num=1
panel=5e-2

def Eriri(y,calc_y):
    loss=0
    for i in range(0,len(y)):
        loss+=(y[i]-calc_y[i])**2
    return 0.5*loss/len(calc_y)
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
    MatB=[]
    for i in range(0,order+1):
        ty=0.0
        for j in range(0,len(xa)):
            ty+=ya[j]*(xa[j]**i)
        MatB.append(ty)
    MatB=np.array(MatB)
    NewMat=np.linalg.solve(MatA,MatB)
    xxa=np.arange(-1,1,0.01)
    x_new=xxa/regulaztion_num
    yya=[]
    for i in range(0,len(xxa)):
        y=0
        for j in range(0,order+1):
            dy=(x_new[i]**j)
            dy*=NewMat[j]
            y+=dy
        yya.append(y)
    # eriri=Eriri(ya,yya)
    # print(eriri)
    ax.plot(xxa,yya,color='g',linestyle='-',marker="")
    ax.legend()
    plt.show()
