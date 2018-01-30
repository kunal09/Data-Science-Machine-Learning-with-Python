#using python 2.7.12

#Part 1 Question 1

import pandas as pd
import numpy as np
import sys

train_file=sys.argv[1]
test_file=sys.argv[2]

#learning rate
n=sys.argv[3]

#taking inputs
data=np.genfromtxt(train_file, dtype='string', delimiter=',')
#removing the first line
data=data[1:,]
#changing its data type to float
data=data.astype(np.float)

#normalizing data
data=np.divide( data-data.mean(0) , np.std(data,axis=0) )

#our variable x		Nx19
x=np.asmatrix(data[:,2:])	

#prices			Nx1
y=data[:,1]
y=np.asmatrix(y)

#weights		19x1
w=np.zeros(19)
w=np.asmatrix(w)

#no of times the iterations is run
T=2500

for i in range(T):
	fn=np.divide( w.dot(x.transpose()) , len(x) )
	w=w-((fn-y).dot(x))*float(n)
#our function is w.dot(x.transpose()) here

#outputs
output=np.empty([4000,2])
output=np.asmatrix(output)

#taking test inputs
test=np.genfromtxt(test_file, dtype='string', delimiter=',')
test=test[1:,:]			#removing the header
test=test.astype(np.float)	#making the file as float
test=np.asmatrix(test)		#test is a matrix now

#transferring ids
output[:,0]=test[:,0]

#removing ids from our testing data
test=test[:,1:]

#normalizing data
test=np.divide( test-test.mean(0) , np.std(test,axis=0) )

#predicting prices
result=test.dot(w.transpose())
result=np.absolute(result)

#sending data to our output file
output[:,1]=result[:,0]
df=pd.DataFrame(output,columns=["id","price"])
df.to_csv('output.csv',index=False)







