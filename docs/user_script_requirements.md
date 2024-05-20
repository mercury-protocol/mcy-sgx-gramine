# User script requirements

### • It shall be named `user_script.py`.
### • It shall define `N_EPOCHS` integer which specifies the number of training epochs.
### • It shall define `create_model` function which shall return the initialized model.
### • It shall define `create_optimizer` function which shall return the initialized optimizer.
### • It shall define `create_data_loader` function which shall return the initialized data loader.
### • It shall define `create_args_from_data_loader` function which can optionally return objects derived from initialized data loader to be used as extra arguments in `train_batch`.
The function shall return the extra arguments in the same order as `train_batch` will expect them.
If you don't need any extra arguments for `train_batch`, you can just leave the function blank:
  ```
  def create_args_from_data_loader(data_loader: DataLoader):
      pass
  ```
### • It shall define the `train_batch` function which is the custom training logic defined by the user to be performed on each data batch.<br>
The function shall have 3 arguments: `batch`, `model` and `optimizer`, 
where `batch` is the iteration variable of the data loader.
  ```
  def train_batch(batch, model, optimizer):
      <training algorithm implementation>
  ```
Plus it can have any user defined arguments which shall be able to be created with `create_args_from_data_loader`.
  ```
  def create_args_from_data_loader(data_loader: DataLoader):
      <creating scheduler and progress bar objects>
      return scheduler, progress bar
  
  def train_batch(batch, model, optimizer, scheduler, progress_bar):
      <training algorithm implementation>
  ```
