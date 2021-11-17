# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '/Users/aj/Documents/DeepLearning/CIFAR10/',
	"depth": 2,
	"learning_rate": 0.01,
	"momentum" : 0.9,
	"network_size" : 7,
	"num_classes" : 10,
	"first_num_filters" : 16,
	# ...
}

training_configs = {
	"learning_rate": 0.01,
	"max_epochs": 10,
	"batch_size": 64,
	"save_interval": 5,


	# ...
}

### END CODE HERE