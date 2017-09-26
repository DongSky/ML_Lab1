import argparse
import math
import numpy as np
import os
import random
mul=1
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Data Generate")
    parser.add_argument('-n',action='store',dest='train_size',default=1000,type=int)
    parser.add_argument('-t',action='store',dest='test_size',default=200,type=int)
    parser.add_argument('-a',action='store',dest='arg_a',default=4.0,type=float)
    parser.add_argument('-b',action='store',dest='arg_b',default=0.0,type=float)
    parser.add_argument('-m',action='store',dest='arg_mu',default=0.0,type=float)
    parser.add_argument('-s',action='store',dest='arg_sigma',default=0.1,type=float)
    result=parser.parse_args()
    def f(x,a=result.arg_a,b=result.arg_b,mu=result.arg_mu,sigma=result.arg_sigma):
        return math.sin(a*(x+b))+np.random.normal(mu,sigma)
    fi=open("train.txt","w")
    for i in range(result.train_size):
        x=mul*random.random()
        if(random.randint(0,1)==1):
            x=0-x
        fi.write(str(x)+","+str(f(x))+"\n")
    fi.close()
    fi=open("test.txt","w")
    for i in range(result.test_size):
        x=mul*random.random()
        if(random.randint(0,1)==1):
            x=0-x
        fi.write(str(x)+","+str(f(x))+"\n")
    fi.close()
