import tarfile
import numpy as np
import sys
import os
import pickle
from scipy import ndimage
"""

    this piece of code extract data in the .tar.gz file,
    read the float value of each image, transform each image into a 28 * 28 matrix
    select 8000 images to form our training set (due to the machine restrictions)
    select about 2000 images to form our testing set
    each character (from A to J) data forms a folder(training data and test data)
    ******************************************************
    After executed this piece of code, there will be a file named 'notMnist.pickle'
    in the dir_root, which is a diction type object being serialized.
    the file contains training set, valid set, and test set.
    you can easily use the data set by pickle.load without considering much about
    the data process operation

    learning from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb
"""

dir_root = '/home/alexsun/ML/data_center/notMnist'
global data_root


# extract files in target rar file
def maybe_extract(file_name, force=False):
    global data_root
    data_root = dir_root # extract to current dir
    file_name = os.path.join(dir_root, file_name)
    head_name = os.path.splitext(os.path.splitext(file_name)[0])[0] # remove .tar.gz
    root = os.path.join(dir_root, head_name)
    # print(root)
    if os.path.isdir(root) and not force:
        print('%s already present, skipping extraction of %s.' %(root, file_name))
    else:
        print('extracting data for %s, wait for a while...' % file_name)
        tar = tarfile.open(file_name)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    print('extracted tar files')
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, d))
                   ]
    # print(data_folders)
    return data_folders

image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images, max_num_images=8000):
    image_files = os.listdir(folder)
    # max_num_images = len(image_files)
    data_set = np.ndarray(shape=(max_num_images, image_size, image_size), dtype=np.float32)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            # read image data  into a float32 matrix
            image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
            # print(image_data.shape)
            if image_data.shape != (image_size, image_size):
                raise Exception('unexpected image size %s ' % (str(image_data.shape)))
            data_set[0:num_images, : , : ] = image_data
            num_images = num_images + 1
            if num_images > max_num_images:
                break
            else:
                if num_images % 100 == 0:
                    print('loading data, %.2f%% finished...' % (num_images * 100.0 / max_num_images))
        except IOError as e:
            print('Could not read file name: %s, we just skipping it..' % image_file)
    data_set = data_set[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full data set tensor:', data_set.shape)
    print('Mean:', np.mean(data_set))
    print('Standard deviation:', np.std(data_set))
    return data_set

train_filename = 'notMNIST_large.tar.gz'
test_filename = 'notMNIST_small.tar.gz'
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


def maybe_pickle(data_folders, force = False):
    data_set_names = []
    for df in data_folders:
        set_filename = df + '.pickle'
        data_set_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('file: %s already exists, just ignoring it..' % set_filename)
        else:
            print('picking %s...' % set_filename)
            data_set = load_letter(df, min_num_images=100)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to %s ' % set_filename, e)
    return data_set_names

train_data = maybe_pickle(train_folders)
test_data = maybe_pickle(test_folders)


def make_array(nb_row, image_size):
    if nb_row:
        data_set = np.ndarray((nb_row, image_size, image_size), dtype=np.float32)
        labels = np.ndarray((nb_row, image_size, image_size), dtype=np.float32)
    else:
        data_set, labels = None, None
    return data_set, labels


def merge_data_sets(pickle_files, training_size, valid_size=0):
    """
    make the .pickle file into training_data set, training_data labels
    validation_data set, validation_labels
    or testing_data set, testing data labels
    :return:
    """
    num_classes = len(pickle_files)
    valid_data_set, valid_data_label = make_array(valid_size, image_size)
    training_data_set, training_data_label = make_array(training_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = training_size // num_classes
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for (label, file) in enumerate(pickle_files):
        try:
            with open(file, 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_data_set is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_data_set[start_v:end_v, :, :] = valid_letter
                    valid_data_label[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                training_data_set[start_t:end_t, :, :] = train_letter
                training_data_label[start_t: end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('unable to read data from: ', file, e)
            raise e
    return valid_data_set,valid_data_label, training_data_set, training_data_label

training_size = 7000
valid_size = 1000
test_size = 1500
valid_data_set, valid_data_label, training_data_set, training_data_label = \
    merge_data_sets(train_data, training_size, valid_size)
_, _, test_data_set, test_data_label = merge_data_sets(test_data, test_size)
print('training_data_set: shape: %s, training_data_label: shape %s'
      %(str(training_data_set.shape), str(training_data_label.shape)))
print('valid_data_set: shape: %s, valid_data_label: shape %s'
      %(str(valid_data_set.shape), str(valid_data_label.shape)))
print('test_data_set: shape: %s, test_data_label: shape %s'
      %(str(test_data_set.shape), str(test_data_label.shape)))

def randomize(data_set, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_data_set = data_set[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_data_set, shuffled_labels

training_data_set, training_data_label = randomize(training_data_set, training_data_label)
test_data_set, test_data_label = randomize(test_data_set, test_data_label)
valid_data_set, valid_data_label = randomize(valid_data_set, valid_data_label)


def save_data_set():
    """
    the most important function of this py module
    save training_data_set, training_data_label, test_data_set,test_data_label
    valid_data_set, valid_data_label into a .pickle file in local disk
    the overview data structure is a diction
    :return None:
    """

    pickle_file = os.path.join(dir_root, 'notMNIST.pickle')
    if os.path.exists(pickle_file):
        print('%s already exist, ignoring saving...' % pickle_file)
        return None
    try:
        f = open(pickle_file, 'wb')
        save = {
            'training_data_set': training_data_set,
            'train_data_label': training_data_label,
            'valid_data_set': valid_data_set,
            'valid_data_label': valid_data_label,
            'test_data_set': test_data_set,
            'test_data_label': test_data_label,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

save_data_set()
if __name__ == '__main__':
    print('!')
    # print(train_data)
    # print(test_data)