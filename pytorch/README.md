# User script requirements

### • It shall be named `user_script.py`.
### • It shall create an integer named `N_EPOCHS` which specifies the number of training epochs.
### • It shall create a `NetworkFactory` object named `network_factory`.
  ```
  from required_utils import NetworkFactory
  
  class MyNeuralNetwork:
    <class implementation>
  
  network_factory = NetworkFactory(MyNeuralNetwork)
  ```
### • It shall create an `OptimizerFactory` object named `optimizer_factory`.
  If your optimizer would be created like this:<br>
  `my_optimizer = optim.SGD(network_parameters, lr=LEARNING_RATE, momentum=MOMENTUM)`<br>
  then the factory instantiation should look like this, taking all args and kwargs that would create the optimizer:
  ```
  from required_utils import OptimizerFactory
  
  optimizer_factory = OptimizerFactory(optim.SGD, lr=LEARNING_RATE, momentum=MOMENTUM)
  ```
### • It shall create a `DataSetFactory` object.
  If your data set would be created like this:<br>
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
  _(Note: the naming of this object is custom as it is only required for creating the data_loader_factory.)_
### • It shall create a `DataLoaderFactory` object named `data_loader_factory`.
  If your data loader would be created like this:<br>
  ```
  my_data_loader = MyDataLoader(
    my_dataset,
    batch_size=64,
    shuffle=True
  )
  ```
  then the factory instantiation should look like this, taking all args and kwargs that would create the data loader:
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
### • It shall define the `train_batch` function which is the custom training logic defined by the user to be performed on each data batch.<br>
The function takes 4 parameters: `data`, `target`, `network` and `optimizer`, 
where `data` and `target` are the iterations of the data loader.
  ```
  def train_batch(data, target, network, optimizer):
    <training algorithm implementation>
  ```