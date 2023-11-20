import numpy as np
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from scipy.spatial import Voronoi, voronoi_plot_2d
import torch
from torch.autograd import grad
import torch.nn.functional as F

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

class AutoEncoder():
    def __init__(self, dims, activation_list=None):
        self.layers = len(dims)-1
        self.params = {}

        for l in range(self.layers):
            self.params["W"+str(l+1)] = 0.01*torch.randn(dims[l], dims[l+1], requires_grad=True, dtype=torch.float32)
            self.params["b"+str(l+1)] = torch.zeros((dims[l+1], 1), requires_grad=True, dtype=torch.float32)

    def forward(self, x):
        x = torch.mm(self.params["W1"].T, x.T) + self.params["b1"]
        x = relu(x)
        x = torch.mm(self.params["W2"].T, x) + self.params["b2"]
        x = relu(x)
        x = torch.mm(self.params["W3"].T, x) + self.params["b3"]
        x = relu(x)
        x = torch.mm(self.params["W4"].T, x) + self.params["b4"]
        x = relu(x)
        x = torch.mm(self.params["W5"].T, x) + self.params["b5"]
        x = relu(x)
        x = torch.mm(self.params["W6"].T, x) + self.params["b6"]
        return x

    def test(self, x):
        x = torch.mm(self.params["W1"].T, x.T) + self.params["b1"]
        x = relu(x)
        x = torch.mm(self.params["W2"].T, x) + self.params["b2"]
        x = relu(x)
        x = torch.mm(self.params["W3"].T, x) + self.params["b3"]
        return x

def relu(z):
    A = torch.clamp(z, min=0.0, max=float('inf'))
    return A

def train(model, x, labels, epochs=20, l_rate=0.1, seed=1):
    cost = []
    torch.manual_seed(seed)

    for i in range(epochs):
        logits = model.forward(x)

        loss = F.mse_loss(logits, labels)
        cost.append(loss.detach())

        if not i%20:
            print('epoch: %02d | Loss: %.5f' % ((i+1), loss))

        gradients = grad(loss, list(model.params.values()))
        k = 1
        with torch.no_grad():
            for j in range(0, model.layers, 2):
                model.params["W"+str(k)] += -l_rate*gradients[j]
                model.params["b"+str(k)] += -l_rate*gradients[j+1]
                k += 1
    return cost

def getSum(cm):
    sum = 0
    for i in range(10):
        sum += cm[i][i]
    return sum

def voronoi_split(x, y):
    out = [[0, 0] for i in range(10)]
    count = [0 for i in range(10)]
    
    for i in range(len(y)):
        out[y[i]][0] += x[y[i]][0]
        out[y[i]][1] += x[y[i]][1]
        count[y[i]] += 1
    for i in range(10):
        out[i][0] /= count[i]
        out[i][1] /= count[i]

    return out

mndl = MnistDataloader(train_images, train_labels, test_images, test_labels)
(x_train, y_train), (x_test, y_test) = mndl.load_data()

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(x_train, y_train)

pred_y = knn.predict(x_test)
cm = confusion_matrix(y_test, pred_y)
s = getSum(cm)
print("Accuracy: ", s/10000)

plt.imshow(cm, cmap = 'inferno', interpolation='nearest')
plt.xlabel('preds')
plt.ylabel('vals')
plt.show()
plt.clf()

dimensions = [1, 2, 5, 10, 100]
pca = decomposition.PCA()
pca.n_components = 2
pca_transform = pca.fit_transform(x_test)
pca_train = pca.fit_transform(x_train)
plot = voronoi_split(pca_train, y_train)
v = Voronoi(plot)
fig = voronoi_plot_2d(v)

for i in range(0, len(y_test), 10):
    if y_test[i] == 0:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='red')
    elif y_test[i] == 1:
       plt.scatter(pca_transform[i][0], pca_transform[i][1], color='orange')
    elif y_test[i] == 2:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='yellow')
    elif y_test[i] == 3:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='green')
    elif y_test[i] == 4:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='blue')
    elif y_test[i] == 5:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='purple')
    elif y_test[i] == 6:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='pink')
    elif y_test[i] == 7:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='olive')
    elif y_test[i] == 8:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='black')
    elif y_test[i] == 9:
        plt.scatter(pca_transform[i][0], pca_transform[i][1], color='cyan')
plt.show()
plt.clf()

pca.n_components = 2
pca_transform = pca.fit_transform(x_train)
pca_test = pca.fit_transform(x_test)

for i in range(len(dimensions)):
    knn = KNeighborsClassifier(n_neighbors=dimensions[i])
    knn.fit(pca_transform, y_train)
    pred_y = knn.predict(pca_test)
    cm = confusion_matrix(y_test, pred_y)
    s = getSum(cm)
    print("Accuracy[", dimensions[i],"]: ", s/10000)

    plt.imshow(cm, cmap = 'inferno', interpolation='nearest')
    plt.xlabel('preds')
    plt.ylabel('vals')
    plt.show()
    plt.clf()

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

seed = 1
torch.manual_seed(seed)
epochs = 500
l_rate = 0.05

dims = [784, 300, 100, 2, 100, 300, 784]

model = AutoEncoder(dims)
cost = train(model, x_train, y_train)

ae_dataset_test = model.test(x_test).detach().numpy()
ae_dataset_train = model.test(x_train).detach().numpy()

for i in range(len(dimensions)):
    knn = KNeighborsClassifier(n_neighbors=dimensions[i])
    knn.fit(ae_dataset_train.T, y_train)
    pred_y = knn.predict(ae_dataset_test.T)

    cm = confusion_matrix(y_test, pred_y)
    s = getSum(cm)
    print("Accuracy[", dimensions[i],"]: ", s/10000)

    plt.imshow(cm, cmap = 'inferno', interpolation='nearest')
    plt.xlabel('preds')
    plt.ylabel('vals')
    plt.show()
    plt.clf()
