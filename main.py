import numpy as np
np.set_printoptions(suppress=False)
from math import e
def sigmoidfun(val):
    sigmoidvalue=1/(1+e**(-val))
    return sigmoidvalue
def determineoutputvalue(sigmoidvalue):
    if sigmoidvalue<0.5:
        return 1
    else:
        return 2
def findcost(cost,outputs,class_label): 
    for i,j in zip(outputs,class_label):
        cost=cost+(j-i)**2
    return cost

def training(input_vector,weight_vector,class_label,bias_vector,weight_vector2,bias2):
    weight_vector_transpose=np.transpose(weight_vector)
    perceptrons_value=np.dot(input_vector,weight_vector)
    persumbias=np.add(perceptrons_value,bias_vector)
    ans=np.zeros((1,16))
    for j in range(0,16):
        ans[0,j] =sigmoidfun(persumbias[0,j])
    ans=np.transpose(ans)
    result=np.dot(weight_vector2,ans)
    result=result+bias2
    return sigmoidfun(result)

with open('iris dataset.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    input_vector = np.loadtxt(lines, delimiter=',', skiprows=1)
    class_label=np.array(input_vector[:,4])
    input_vector=np.delete(input_vector,4,1)
    weight_vector=5*np.random.random((4,16))-3  # initialize weight vector of hidden layer
    bias_vector=np.random.rand(1,16)
    weight_vector2=3*np.random.random((1,16))-2 # initialize weight vector of output layer
    bias2=np.random.rand(1,1)
    weight_vector_transpose=np.transpose(weight_vector) #transpose of a weight vector
    outputs=[]
    for i in input_vector:
        o1=training(i,weight_vector,class_label,bias_vector,weight_vector2,bias2)
        o1=determineoutputvalue(o1)
        outputs.append(o1) 
    cost=0
    totalsamples=len(input_vector)
    cost=findcost(cost,outputs,class_label)
    accuracy=(cost/totalsamples)*100 # calculating accuracy in percentage
    print(accuracy,'%')

    

