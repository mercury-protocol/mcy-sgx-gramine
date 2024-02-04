## Requirements from user script

* it shall be named `user_script.py`
* it shall create an integer named `N_EPOCHS` which is the number of training epochs
* it shall create a `network_factory` object using `required_utils.NetworkFactory`
  * example:  
  ```
  from required_utils import NetworkFactory
  
  class MyNeuralNetwork:
    <class implementation>
  
  network_factory = NetworkFactory(MyNeuralNetwork)
  ```
* it shall create an `optimizer_factory` object using `required_utils.NetworkFactory`
  * if your optimizer would be created like this:<br>
  `my_optimizer = optim.SGD(network_parameters, lr=LEARNING_RATE, momentum=MOMENTUM)`<br>
  then the factory instantiation should look like this, taking all args and kwargs that would create the optimizer:
  ```
  from required_utils import OptimizerFactory
  
  optimizer_factory = OptimizerFactory(optim.SGD, lr=LEARNING_RATE, momentum=MOMENTUM)
  ```
* it shall create a `data_set_factory` object using `required_utils.DataSetFactory`
  * if your data set would be created kie this:<br>
  ```
  my_data_set = MyDataSet(
    "path/to_my/data",
    some_arg,
    some_kwarg="some kwarg value"
  )
  ```
  then the factory instantiation should look like this, taking all args and kwargs that would create the data set, except data path:
  ```
  from required_utils import DataSetFactory
  
  class MyDataSet:
    <class implementation>
  
  data_set_factory = DataSetFactory(
    MyDataSet,
    some_arg,
    some_kwarg="some kwarg value"
  )
  ```
* it shall create a `data_loader_factory` object using `required_utils.DataLoaderFactory`
  * if your data loader would be created kie this:<br>
  ```
  my_data_loader = MyDataLoader(
    my_dataset,
    batch_size=64,
    shuffle=True
  )
  ```
  then the factory instantiation should look like this, taking all args and kwargs that would create the data loader, except data path:
  ```
  from required_utils import DataLoaderFactory
  
  class MyDataLoader:
    <class implementation>
  
  data_loader_factory = DataLoaderFactory(
    MyDataLoader,
    data_set_factory,
    batch_size=64,
    shuffle=True
  )
  ```
* it shall define the `train_batch` function which is the custom training logic defined by the user to be performed on one data batch.<br>
The function takes 4 parameters: `data`, `target`, `network` and `optimizer`, 
where `data` and `target` are the iterations of the data loader.
  ```
  def train_batch(data, target, network, optimizer):
    <training algorithm implementation>
  ```