import numpy as np
data=[]
labels=[]
def KNNClassify(newInput,dataSet,labels,k):
    numSamples=dataSet.shape[0] # the num of row
    diff=np.tile(newInput,(numSamples,1))-dataSet
    squareDiff=diff**2
    squareDist=squareDiff.sum(axis=1)
    distance=squareDist**0.5
    sortedDistIndices=np.argsort(distance)
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    maxCount=0
    for key,value in classCount.items():
        if value>maxCount:
            maxCount=value
            maxIndex=key
    return maxIndex

with open("data.txt") as ifile:
    for line in ifile:
        tokens=line.strip().split(' ')
        data.append([float(num) for num in tokens[:-1]])
        labels.append(tokens[-1])
x=np.array(data)
labels=np.array(labels)
print(x)
print(labels)
test=np.array([1.9,60])
output=KNNClassify(test,x,labels,7)
print(output)

