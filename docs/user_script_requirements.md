# User script requirements

### • It shall be named `user_script.py`.
### • It shall define `N_EPOCHS` integer which specifies the number of training epochs.
### • It shall define `BATCH_SIZE` integer which specifies the batch size of the data loader.
### • It shall define `create_model` function which shall return the initialized model.
### • It shall define `create_optimizer` function which shall return the initialized optimizer.
### • It shall define `create_data_loader` function which shall return the initialized data loader.
### • It shall define `create_extra_training_args` function which can optionally return objects derived from initialized data loader and optimizer to be used as extra arguments in `train_batch`.
The function shall return the extra arguments in the same order as `train_batch` will expect them.
If you don't need any extra arguments for `train_batch`, you can just leave the function blank:
  ```
  def create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer):
      pass
  ```
### • It shall define the `train_batch` function which is the custom training logic defined by the user to be performed on each data batch.<br>
The function shall have 3 arguments: `batch`, `model` and `optimizer`, 
where `batch` is the iteration variable of the data loader. The `optimizer.zero_grad()` shall be performed at the beginning of the function - otherwise the gradient aggregation would not work.
  ```
  def train_batch(batch, model, optimizer):
      <training algorithm implementation>
  ```
Plus it can have any user defined arguments which shall be created with `create_extra_training_args`. 
This is for creating objects for the lifetime of the full training cycle - for example a learning rate scheduler which needs to know the number of batches and the current batch.
  ```
  def create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer):
      <creating scheduler and progress bar objects>
      return scheduler, progress bar
  
  def train_batch(batch, model, optimizer, scheduler, progress_bar):
      <training algorithm implementation>
  ```
