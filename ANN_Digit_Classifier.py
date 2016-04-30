import numpy as np
import scipy.io
import sklearn.preprocessing
import sklearn.cross_validation
import NeuralNetwork as nn
import pickle
import matplotlib.pyplot as plt


def fix_labs(val):
    label_vec = np.zeros(10)
    label_vec[val] = 1
    return label_vec


def centroid_calc(N, imgs):
    imgs = np.array(imgs.reshape((1, len(imgs))))
    imgs.reshape([N, 28, 28])
    imgtotalsum = np.sum(np.sum(imgs, axis=0), axis=0)

    indices = np.arange(28)
    X, Y = np.meshgrid(indices, indices)
    X = X.reshape((1, 784))
    Y = Y.reshape((1, 784))
    centroidx = np.sum(np.sum(imgs * X, axis=0), axis=0) / imgtotalsum
    centroidy = np.sum(np.sum(imgs * Y, axis=0), axis=0) / imgtotalsum
    return centroidx, centroidy


def custom_feature_extractor(imagetrue):
    featureslist = np.array([item for sublist in imagetrue for item in sublist])
    ctr_x, ctr_y = centroid_calc(1, featureslist)
    h = np.size(imagetrue, 0)
    w = np.size(imagetrue, 1)
    featureslist = np.append(featureslist,
                             [np.average(imagetrue[h / 2:, w / 2:]), np.average(imagetrue[h / 2:, :w / 2]),
                              np.average(imagetrue[:h / 2, w / 2:]), np.average(imagetrue[:h / 2, :w / 2])])

    topclosed = False
    prevrow = (False, False, [], [], [])
    numholes = 0
    edgecount = 0
    for row in imagetrue:
        for i in range(np.size(row) - 1):
            if row[i] != 0 and row[i + 1] == 0:
                edgecount += 1
            elif row[i] == 0 and row[i + 1] != 0:
                edgecount += 1

        nonzero_elems = np.nonzero(row)[0]
        if len(nonzero_elems) > 0:
            separation = np.ediff1d(nonzero_elems) - 1
            nonzerodiffs = np.nonzero(separation)[0]
            startnonzero = nonzero_elems[0]
            endnonzero = nonzero_elems[-1]
            if len(nonzerodiffs) > 0:  # has gaps
                gaps_in_row = []
                for block in nonzerodiffs:
                    gap_start_index = nonzero_elems[block]
                    gap_end_index = nonzero_elems[block] + separation[block] + 1
                    gaps_in_row.append((gap_start_index, gap_end_index))

                # do stuff here, that has gaps of 0pixels, figure out how to detect holes
                if prevrow[0]:
                    if prevrow[1]:  # prevrow has Gaps, this row has gaps
                        for prevGap in prevrow[2]:
                            leakfromtop = shortcircuit = True
                            for ii in range(len(gaps_in_row)):
                                for p in range(gaps_in_row[ii][0], gaps_in_row[ii][1] + 1):
                                    shortcircuit = shortcircuit and prevrow[4][p] != 0
                                    if not shortcircuit:
                                        break
                                if shortcircuit:
                                    topclosed = True

                                if prevGap[0] >= startnonzero and prevGap[1] <= endnonzero:
                                    leakfromtop = False
                                    break
                            if leakfromtop:
                                topclosed = False
                                leakfromtop = False
                    else:  # prevrow no gaps, this row has gaps
                        for gap in gaps_in_row:
                            if not prevrow[3][0] <= gap[0] + 1 or not prevrow[3][1] >= gap[1] - 1:
                                topclosed = False

                else:  # zero pixels filled row
                    topclosed = False
                prevrow = (True, True, gaps_in_row, (startnonzero, endnonzero), row)
            else:  # only stuff here has no zero gaps in it
                if not prevrow[0]:  # prev row was all zeros and this row is all solid
                    topclosed = True
                else:  # prev row not all zeros
                    if prevrow[1]:  # prevrow has gaps, this row solid
                        for gap in prevrow[2]:
                            if startnonzero - 1 <= gap[0] and gap[1] <= endnonzero + 1:
                                if topclosed:
                                    numholes += 1
                    else:  # prevrow solid, this row solid
                        topclosed = True
                prevrow = (True, False, [], (startnonzero, endnonzero), row)
        else:  # row is all zeros
            prevrow = (False, False, [], (), row)

    binary = [0, 0, 0]
    if numholes > 2:
        numholes = 2
    binary[numholes] = 1

    return np.append(featureslist, binary + [edgecount, ctr_x, ctr_y])


mat = scipy.io.loadmat("dataset/train.mat")
train_images = np.array(mat['train_images'])
# testing = np.array(mat['test_images'])
train_labels = np.array(mat["train_labels"])
numTrainImages = np.size(train_images, 2)
# numTestImages = np.size(testing, 2)

numSamples = numTrainImages
print("total training samples = " + str(numTrainImages))
print("number samples using = " + str(numSamples))

images = np.matrix(np.empty((numSamples, len(custom_feature_extractor(train_images[:, :, 0])))))
# test_images = np.matrix(np.empty((numSamples, len(custom_feature_extractor(testing[:, :, 0])))))
for i in range(numSamples):
    images[i] = np.matrix(custom_feature_extractor(train_images[:, :, i]))
labels_matrix = np.matrix([fix_labs(item) for item in train_labels])

persistentStore = {"images": images, "labels_matrix": labels_matrix}  # "test_images":test_images}
with open('store training structures.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(persistentStore, f, pickle.HIGHEST_PROTOCOL)

hidden = 100
batch_size = 10
learning_rate = .1
loss_func = "cross_entropy"

with open('store training structures.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)
    labels_matrix, images = data["labels_matrix"], data["images"]

neural_net = nn.NeuralNetwork(hidden=hidden)

images = np.matrix(sklearn.preprocessing.normalize(images, norm="l2", axis=1))

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(images, labels_matrix, test_size=0)
print("training = " + str(np.shape(X_train)))
print("labels = " + str(np.shape(y_train)))

plot_x = np.empty((0, 0))
plot_y = np.empty((0, 0))
plot_y2 = np.empty((0, 0))

V, W, plot_x, plot_y, plot_y2 = neural_net.train(X_train, y_train, plot_x, plot_y, plot_y2, learning_rate=learning_rate,
                                                 batch_size=batch_size)

mat = scipy.io.loadmat("dataset/test.mat")
test_images = np.array(mat['test_images'])
images = np.matrix(np.empty((test_images.shape[0], len(custom_feature_extractor(test_images[0, :, :])))))

for i in range(test_images.shape[0]):
    images[i] = np.matrix(custom_feature_extractor(test_images[i, :, :]))

images = np.matrix(sklearn.preprocessing.normalize(images, norm="l2", axis=1))

persistentStore = {"V": V, "W": W, "batch_size": batch_size, "learning_rate": learning_rate,
                   "hidden_layers": hidden, "lossfunction": loss_func, "plot_x": plot_x, "plot_y": plot_y,
                   "plot_y2": plot_y2, "images": images}

with open('FinishedNN.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(persistentStore, f, pickle.HIGHEST_PROTOCOL)

# with open('FinishedNN.pickle', 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     data = pickle.load(f)
#     V, W, hidden, plot_x, plot_y, plot_y2, images = data["V"], data["W"], data["hidden_layers"], data["plot_x"], data[
#         "plot_y"], data["plot_y2"], data["images"]

neural_net = nn.NeuralNetwork(hidden=hidden)
predicted = np.argmax(neural_net.predict(V, W, images).getT(), axis=1)

f = open("output.csv", 'w')
f.write("Id,Category\n")
for i in range(predicted.shape[0]):
    f.write(str(i + 1) + "," + str(predicted[i][0, 0]) + "\n")
print("DONE!")

plt.plot(plot_x, plot_y, "r-")
#plt.plot(plot_x, plot_y2)
plt.ylabel('Accuracy Rate')
plt.xlabel('Iterations Number')
plt.show()

# Z_train = neural_net.predict(V, W, X_train)
# Z_test = neural_net.predict(V, W, X_test)
#
# train_err = (sum(np.argmax(y_train, axis=1) == np.argmax(Z_train.getT(), axis=1))[0, 0]) / (y_train.shape[0])
# print("Training Accuracy is: {0}".format(train_err))
#
# test_err = (sum(np.argmax(y_test, axis=1) == np.argmax(Z_test.getT(), axis=1))[0, 0]) / (y_test.shape[0])
# print("Testing Accuracy is: {0}".format(test_err))
