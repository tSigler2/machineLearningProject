import numpy as np
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from scipy.spatial import Voronoi, voronoi_plot_2d

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
'''
class autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9))

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 784),
            torch.nn.Sigmoid())
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def autoencode(self, x):
        pass
'''
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

for i in range(len(dimensions)):
    pca.n_components = dimensions[i]
    pca_transform = pca.fit_transform(x_train)
    pca_test = pca.fit_transform(x_test)
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
