import pickle
import os
import matplotlib.pyplot as plt

dir_root = '/home/alexsun/ML/data_center/notMnist'
with open(os.path.join(dir_root, 'notMNIST_small/A.pickle'), 'rb') as f:
    test_a = pickle.load(f)

sample = test_a[11]
for i in range(28):
    plt.plot([i for i in range(28)], sample[i], 'ro')
plt.show()