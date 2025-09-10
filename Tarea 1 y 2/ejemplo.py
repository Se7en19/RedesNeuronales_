import mnist_loader
import network
import pickle

# ----------------------
# - read the input data:
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)



# ---------------------
# - network.py example:
import network


net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)