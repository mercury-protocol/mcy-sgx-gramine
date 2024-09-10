# mcy-sgx-gramine

This is a package used in [Mercury Protocol](https://mercuryprotocol.netlify.app)'s [vulkan](https://github.com/mercury-protocol/vulkan) repository for training AI models distributed.

## Usage:

Each instance has a role: WATCHER or LEADER. Watchers do the batch training and the leader does the gradient aggregation.

The user who wants to train an AI model has to write the script in a file called `user_script.py`. This file is used by this component to perform the training. To see how it should be written check `docs/user_script_requirements.md` and `docs/user_script_template.py`.
