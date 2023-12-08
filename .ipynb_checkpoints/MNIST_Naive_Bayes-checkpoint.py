import numpy as np
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

in_path = './archive'
train_images = join(in_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
train_labels = join(in_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images = join(in_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels = join(in_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

class MnistDataloader(object):
    def __init__(self, training_images, training_labels, test_images, test_labels):
        self.training_images = training_images
        self.training_labels = training_labels
        self.test_images = test_images
        self.test_labels = test_labels
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images, self.training_labels)
        x_test, y_test = self.read_images_labels(self.test_images, self.test_labels)
        return (x_train, y_train),(x_test, y_test)

def getSum(cm):
    sum = 0
    for i in range(10):
        sum += cm[i][i]
    return sum

def getMeans(xList, yList):
    ret_matrix = []
    aux_matrix = []
    
    for i in range(10):
        for j in range(len(xList)):
            if yList[j] == i:
                aux_matrix.append(xList[i])
        ret_matrix.append(np.mean(np.array(aux_matrix), axis=0).reshape(28, 28))
        aux_matrix = []
    
    return np.array(ret_matrix)

def getVar(xList, yList):
    ret_matrix = []
    aux_matrix = []
    
    for i in range(10):
        for j in range(len(xList)):
            if yList[j] == i:
                aux_matrix.append(xList[i])
        ret_matrix.append(np.var(np.array(aux_matrix), axis=0).reshape(28, 28))
        aux_matrix = []
    return np.array(ret_matrix)



mndl = MnistDataloader(train_images, train_labels, test_images, test_labels)
(x_train, y_train), (x_test, y_test) = mndl.load_data()

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

model = GaussianNB()
nbg = model.fit(x_train, y_train)
pred_y = nbg.predict(x_test)
cm = confusion_matrix(y_test, pred_y)
s = getSum(cm)
print("Accuracy: ", s/10000)

plt.imshow(cm, cmap = 'inferno', interpolation='nearest')
plt.xlabel('preds')
plt.ylabel('vals')
plt.show()
plt.clf()

var_matrix = getVar(x_test, y_test)
mean_matrix = getMeans(x_test, y_test)

for i in range(10):
    plt.imshow(mean_matrix[i], cmap='inferno', interpolation='nearest')
    plt.title(str(i))
    plt.show()
    plt.clf()
    plt.title(str(i))
    plt.imshow(var_matrix[i], cmap='inferno', interpolation='nearest')
    plt.show()
    plt.clf()
