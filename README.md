# Neural-Net
suuper simple neural net implementation for cs 189 (current test accuracy 0.96340; training accuracy 0.97345).... Will work on making better

### Things to improve/add:
  - More generalized backpropagation implementation. Right now just coded up as the closed form gradient with chain rule and only for cross entropy and mean squared losses.
  - L2 Regularization on the weights to reduce overfitting and stuff
  - Support Multiple Hidden Layers, and add more paramaters to make more customizable
  - Replace Sigmoid with 2*Sigmoid(x) - 1 ... Apparently converges faster than normal sigmoid activation function
  - Don't use the same learning rate for different layers of the neural net! Bottom Layers should have bigger epsilon and upper layers should have smaller epsilon
  - Shuffle Data not randomly but in a way that increases the learning (i.e. rare classes more often, no sequence of the same classes)
  - Maybe add a simple line search to get a better gradient descent algorithm
  - Try Differnt activation functions (ReLU, etc...)
