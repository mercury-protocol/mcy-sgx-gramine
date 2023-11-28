## Requirements from user script

* it shall be named `user_script.py`
* it shall create a `network_factory` object using `required_utils.NetworkFactory`
  * example:  
  ```
  from required_utils import NetworkFactory
  
  class MyNeuralNetwork:
    <class implementation>
  
  network_factory = NetworkFactory(MyNeuralNetwork)
  ```
* it shall create an 'optimizer_factory' object using required_utils 'required_utils.NetworkFactory'
  * if your optimizer is created like this:<br>
  `my_optimizer = optim.SGD(network_parameters, lr=LEARNING_RATE, momentum=MOMENTUM)`<br>
  then the factory instantiation should look like this:
  ```
  from required_utils import OptimizerFactory
  
  optimizer_factory = OptimizerFactory(optim.SGD, lr=LEARNING_RATE, momentum=MOMENTUM)
  ```
* it shall define a train function which takes 2 parameters: network and optimizer. This is where the training algorithm shall be implemented.
  ```
  def train(network, optimizer):
    <training algorithm implementation>
  ```