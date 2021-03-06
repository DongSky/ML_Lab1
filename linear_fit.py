import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
#some essential parameters
eps=1e-8
alpha=3e-2
train_size=0
regulaztion_num=1
panel=1
#this is the loss function
def Eriri(y,calc_y):
    loss=0
    for i in range(0,len(y)):
        loss+=(y[i]-calc_y[i])**2
    return 0.5*loss/len(calc_y)
#main procedure
if __name__=="__main__":
    #generate argument list, users can set the order of the polynomial before the program starts, default is 9
    parser=argparse.ArgumentParser(description="Linear Curve Fit")
    parser.add_argument('-o',action="store",dest="order",default=9,type=int)
    result=parser.parse_args()
    order=result.order
    #initialize matplotlib to display the curve
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #read the training data from file
    xa=[]
    ya=[]
    fi=open("train.txt","r")
    lines=fi.readlines()
    for line in lines:
        piece=line.split(",")
        xa.append(float(piece[0]))
        ya.append(float(piece[1]))
    ax.plot(xa,ya,color='m',linestyle='',marker='.')
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
    MatA=[]
    #calculate X.T * X matrix(named as MatA)
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
    #the parameter w(named as NewMat) can be calculated by inv(X.T * X) * (X.T * Y) when X.T*X is positive definite, else it should be calculated as a similar solution
    NewMat=np.linalg.solve(MatA,MatB)
    print(NewMat)
    #draw the curve in matplotlib using a lot of points
    xxa=np.arange(-1.9,1.9,0.01)
    x_new=xxa/regulaztion_num
    yya=[]
    for i in range(0,len(xxa)):
        y=0
        for j in range(0,order+1):
            dy=(x_new[i]**j)
            dy*=NewMat[j]
            y+=dy
        yya.append(y)

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
        y=0
        for j in range(0,order+1):
            dy=(x_test[i]**j)
            dy*=NewMat[j]
            y+=dy
        y_test_linear.append(y)
    #calculate the loss of test data and output it
    eriri_test_linear=Eriri(y_test,y_test_linear)
    print("linear fit:"+str(eriri_test_linear))
    #show the curve
    ax.plot(xxa,yya,color='g',linestyle='-',marker="")
    ax.legend()
    plt.show()
