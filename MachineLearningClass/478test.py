""" Linear Regression Example """

from __future__ import absolute_import, division, print_function

import tflearn
import numpy as np
file=open("testing.txt", 'r')
Y = []
X = []
predict=[]
for line in file:
   stuff=line.split(',')
   myarray=[]
   myarray.append(float(stuff[0]))
   myarray.append(float(stuff[1]))
   myarray.append(float(stuff[2]))
   myarray.append(float(stuff[3]))
   X.append(myarray)
   #print (stuff[4])
   if stuff[4]=="Iris-setosa\n":
   	Y.append([1, 0 , 0, 0])

   elif stuff[4]=="Iris-virginica\n":
   	Y.append([0, 1, 0, 0])
   else:
    Y.append([0 ,0 , 1, 0])	
# Regression data
#print (X)
#print (Y)
X=np.asarray(X)
Y=np.asarray(Y)
print(X.shape)
print(Y.shape)
# Linear Regression graph
input_ = tflearn.input_data(shape= (64,4))
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([3.2, 3.3, 3.4]))