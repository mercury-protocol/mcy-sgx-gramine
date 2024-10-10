# mcy-dist-ai

This is a package used in [Mercury Protocol](https://mercuryprotocol.netlify.app)'s [vulkan](https://github.com/mercury-protocol/vulkan) repository for training AI models distributed.

## Usage:

Each instance has a role: WATCHER or LEADER. Watchers do the batch training and the leader does the gradient aggregation.

The user who wants to train an AI model has to write the script in a file called `user_script.py` 
and also a `user_requirements.txt` where the dependencies of the `user_script.py` are specified.
This file is used by this component to perform the training. 
To see how it should be written check `docs/user_script_requirements.md` and `docs/user_script_template.py`.

## mcy-split-data:

`mcy-split-data` is command which can be used by the user to split the data into a specified number of partitions and save them as tensors.

args:<br>
`split_into`: the data will be split into this many partitions<br>
`data_path`: the path of the directory which contains the data <br>
`output_dir_path`: the path of the directory where the split tensors will be saved<br>
`user_script_path`: the path of the `user_script.py` file where the `create_data_loader` function is specified by the user<br>

example usage:
```zsh
mcy-split-data 2 path/to/data/dir path/to/output/data/dir path/to/user_script.py
```
