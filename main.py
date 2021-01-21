import numpy as np
import csv

raw=[]
with open('train.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		raw.append(line)

labels=[]
raw2 =[]
for j in range(8000):
	for i in range(1,785):
		raw2.append(float(raw[j][i]))
		labels.append(int(raw[j][0]))

current_label = np.zeros(10)
for i in range(10):
	if i == labels[0]:
		current_label[i] = 1

x=np.zeros([8,784])
for i in range(8):
	x[i]=raw2[784*i:784+784*i]
x=x/255
#x = np.array(raw2[0+i:784+i])/255

class Layers:
	def __init__(self,n_inputs,n_neurons):
		self.weights = np.random.rand(n_inputs, n_neurons)
		self.biases = np.random.rand(1,n_neurons)

	def forward(self,inputs):
		self.outputs = np.dot(inputs,self.weights)+self.biases

	def backprop(self):
		pass

class Sigmoid:
	def forward(self,input):
		self.outputs = 1/(1+np.exp(-input))
	def backward(self,input):
		self.prime = np.exp(-input)/((1+np.exp(-input))**2)

class SoftMax:
	def forward(self,input):
		self.outputs = np.exp(input)/np.sum(np.exp(input), axis=1,keepdims=True)
	def backward(self,input):
		self.prime = np.zeros([np.array(input).shape[0],np.array(input).shape[1],np.array(input).shape[1]])
		for i in range(np.array(input).shape[0]):
			self.prime[i] = np.array(input[i]).T*np.eye(np.array(input).shape[1])-np.dot(np.array(input).T,np.array(input))

class CostMeanSquared:
	def forward(self,input,target,length):
		self.outputs = sum(sum((input - target)**2))/length
	def backward(self,input,target,length):
		self.prime = 2*sum(sum(input - target))/length


'''def SigmoidPrime(x):
	return np.exp(-x)/((1+np.exp(-x)**2))

def SoftMaxPrime(x):
	pass'''


layer1 = Layers(784,16)
activation1 = Sigmoid()

layer2 = Layers(16,16)
activation2 = Sigmoid()

layer3 = Layers(16,10)
activation3 = SoftMax()

cost = CostMeanSquared()

layer1.forward(x)
activation1.forward(layer1.outputs)

layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)

layer3.forward(activation2.outputs)
activation3.forward(layer3.outputs)

cost.forward(activation3.outputs,current_label,10)

y=SoftMax()
y.backward(activation3.outputs)

print(cost.outputs)
print(activation3.outputs)
print(y.prime)


