import torch
from torch import save, load
import os
import time
import argparse

output_model_file = "model_state.pt"

def aggregate_gradients():
    files_in_current_directory = os.listdir()

    gradient_update_files = [file for file in files_in_current_directory if 'gradient_updates' in file]

    state_dict_list = []
    for fname in gradient_update_files:
        with open(fname, 'rb') as f:
            f.seek(0)
            state_dict = torch.load(f)
            state_dict_list.append(state_dict)

    if len(state_dict_list) == 0:
        return

    for params in zip(*[state_dict.values() for state_dict in state_dict_list]):
        param = params[0]
        for param_other in params[1:]:
            param.add_(param_other)

    with open(output_model_file, 'wb') as f:
        save(state_dict_list[0], f)

    for fname in gradient_update_files:
        os.remove(fname)

    if os.path.exists("training_complete"):
         model_path = "trained_model.pth"
         save(state_dict_list[0], model_path)


def aggregate_gradients_and_save_model():
    files_in_current_directory = os.listdir()

    gradient_update_files = [file for file in files_in_current_directory if 'gradient_updates' in file]

    state_dict_list = []
    for fname in gradient_update_files:
        with open(fname, 'rb') as f:
            f.seek(0)
            state_dict = torch.load(f)
            state_dict_list.append(state_dict)

    if len(state_dict_list) == 0:
        return

    for params in zip(*[state_dict.values() for state_dict in state_dict_list]):
        param = params[0]
        for param_other in params[1:]:
            param.add_(param_other)

    model_path = "trained_model.pth"
    save(state_dict_list[0], model_path)

    for fname in gradient_update_files:
        os.remove(fname)
    


def split_dataset():
    # TODO
    return

def parse_worker_nodes_count():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_count", type=int, help="Worker nodes count")
    args = parser.parse_args()
    if args.worker_count is None:
        print("Missing worker nodes count")
        os._exit(1)
    return args.worker_count

def parse_node_num():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_num", type=int, help="Node number")
    args = parser.parse_args()
    if args.node_num is None:
        print("Missing node number")
        os._exit(1)
    return args.node_num

def export_gradients(model, node_num):
    fname = f"gradient_updates_{node_num}.pt"
    with open(fname, 'wb') as f:
            save(model.state_dict(), f)
    return fname

def wait_for_gradient_updates(model, node_num):
    fname = f"gradient_updates_{node_num}.pt"
    
    while not os.path.exists(output_model_file):
                time.sleep(1)

    with open(output_model_file, 'rb') as f:
            model.load_state_dict(load(f)) 
            os.remove(output_model_file)

    return model

def complete_training(model, node_num):
    fname = f"gradient_last_updates_{node_num}.pt"
    
    with open(fname, 'wb') as f:
            save(model.state_dict(), f)

    with open(f"training_complete_{node_num}", 'w'):
        pass