# User script requirements

### • It shall be named `user_script.py`.
### • It shall define `N_EPOCHS` integer which specifies the number of training epochs.
### • It shall define `create_model` function which shall return the initialized model.
### • It shall define `create_optimizer` function which shall return the initialized optimizer.
### • It shall define `create_data_loader` function which shall return the initialized data loader.
### • It shall define the `train_batch` function which is the custom training logic defined by the user to be performed on each data batch.<br>
The function takes 4 parameters: `data`, `target`, `network` and `optimizer`, 
where `data` and `target` are the iterations of the data loader.
  ```
  def train_batch(data, target, network, optimizer):
    <training algorithm implementation>
  ```