import numpy as np
import sklearn.utils
from scipy.special import expit
import pickle


class NeuralNetwork:
    def __init__(self, n_in=784, hidden=200, out=10):
        self.n_hidden = hidden
        self.n_in = n_in
        self.n_out = out

    def predict(self, V, W, images):
        images = np.matrix(np.append(images, np.ones((images.shape[0], 1)), axis=1))
        H = np.matrix(np.append(np.tanh(V * images.getT()), np.ones((1, images.shape[0])), 0))
        return expit(W * H)

    # Train Network Algorithm
    def train(self, images, labels, epochs_for_plot, errRates, lossFunc, learning_rate=0.05, batch_size=1):
        def mean_squared_update(h, z, b, image, image_label):
            # MSE
            delta_w = learning_rate * (z[:, b] - image_label[b, :].getT())
            delta_w = np.multiply(delta_w, np.multiply(z[:, b], (1 - z[:, b])))
            delta_w = np.multiply(delta_w, np.repeat(h[:, b].getT(), 10, axis=0))

            delta_v = np.multiply(learning_rate * (1 - np.square(h[:-1, b])),
                                  sum([W[j, :-1] * (z[j, b] - image_label[b, j]) * (z[j, b] * (1 - z[j, b])) for j in
                                       range(self.n_out)]).getT())
            delta_v = np.multiply(delta_v, np.repeat(image[b, :], self.n_hidden, axis=0))

            return delta_v, delta_w

        def cross_entropy_update(h, z, b, image, image_label):
            # H, and Z are calculated in the forward pass, here we back-propagate to find update rule
            delta_w = learning_rate * (z[:, b] - image_label[b, :].getT())
            delta_w = np.multiply(delta_w, np.matrix(np.repeat(h[:, b].getT(), 10, axis=0)))

            delta_v = np.multiply(learning_rate * (1 - np.square(h[:-1, b])),
                                  sum([W[j, :-1] * (z[j, b] - image_label[b, j]) for j in range(self.n_out)]).getT())
            delta_v = np.multiply(delta_v, np.repeat(image[b, :], self.n_hidden, axis=0))

            return delta_v, delta_w

        self.n_in = np.size(images, 1)  # number of neurons in the input layer = #features, (we don't count the bias)

        # Initialize all weights, V, W at random
        V = np.matrix(np.random.normal(0, 0.01, (self.n_hidden, self.n_in + 1)))
        W = np.matrix(np.random.normal(0, 0.01, (self.n_out, self.n_hidden + 1)))

        epochs, err_rate = 0, 0
        while True:
            images, labels = sklearn.utils.shuffle(images, labels)
            print("epoch# = {0}".format(epochs))
            old_err_rate = err_rate
            ZZZ = self.predict(V, W, images)
            err_rate = (sum(np.argmax(labels, axis=1) == np.argmax(ZZZ.getT(), axis=1))[0, 0]) / (labels.shape[0])
            err_diff = np.math.fabs(err_rate - old_err_rate)
            print("Training Accuracy on epoch {1}: {0}".format(err_rate, epochs))

            if err_diff <= .0000001:
                break

            for i in range((np.size(images, 0) // batch_size) + 1):
                # pick batch data points (x, y) at random from the training set (FOR stochastic GD batch_size = 1)
                image = images[i * batch_size:(i + 1) * batch_size, :]
                image_label = labels[i * batch_size:(i + 1) * batch_size, :]

                if image.size == 0:
                    continue

                # perform forward pass (computing necessary values for gradient descent update)
                # vectorized for efficiency computation of the outputs on all samples
                image = np.matrix(np.append(image, np.ones((image.shape[0], 1)), axis=1))
                H = (np.append(np.tanh(V * image.getT()), np.ones((1, image.shape[0])), 0))
                Z = expit(W * H)

                for b in range(image.shape[0]):
                    del_v, del_w = cross_entropy_update(H, Z, b, image, image_label)
                    V -= del_v
                    W -= del_w

                if labels.shape[0] == 0 or i % 1000 != 0:
                    continue

                useThis = np.matrix(np.append(images, np.ones((images.shape[0], 1)), axis=1))
                H = (np.append(np.tanh(V * useThis.getT()), np.ones((1, useThis.shape[0])), 0))
                Z = expit(W * H)
                ce_loss = .5 * np.sum(np.square(Z[:, :].getT() - labels[:, :]))
                errRates = np.append(errRates,
                                     np.sum(np.argmax(labels, axis=1) == np.argmax(Z.getT(), axis=1)) / (
                                         labels.shape[0]))
                lossFunc = np.append(lossFunc, ce_loss)
                epochs_for_plot = np.append(epochs_for_plot, i)

            if epochs >= 5:
                with open('CE_Loss_epoch_{0}.pickle'.format(epochs), 'wb') as f:
                    pickle.dump({"V": V, "W": W}, f, pickle.HIGHEST_PROTOCOL)

            epochs += 1
            learning_rate *= .1  # anneal the learning rate

        return V, W, epochs_for_plot, errRates, lossFunc
