import torch
from torch import save, load, nn
import os
import time
import argparse

output_model_file = "model_state.pt"

def aggregate_gradients(model, opt):
    files_in_current_directory = os.listdir()

    gradient_update_files = [file for file in files_in_current_directory if 'gradient_updates' in file]
    gradient_updates = [torch.load(file) for file in gradient_update_files]

    if len(gradient_updates) == 0:
         return
    
    avg_aggr_gradients(model=model, gradient_updates=gradient_updates)
            
    opt.step()
    opt.zero_grad()

    with open(output_model_file, 'wb') as f:
        save(model.state_dict(), f)

    for fname in gradient_update_files:
        os.remove(fname)

    model_path = "~/trained_model1.pth"
    save(model.state_dict(), model_path)


def aggregate_gradients_and_save_model(model, opt):
    files_in_current_directory = os.listdir()

    gradient_update_files = [file for file in files_in_current_directory if 'gradient_last_updates' in file]
    gradient_updates = [torch.load(file) for file in gradient_update_files]

    if len(gradient_updates) == 0:
         return
    
    avg_aggr_gradients(model=model, gradient_updates=gradient_updates)
            
    opt.step()
    opt.zero_grad()

    model_path = "~/trained_model.pth"
    save(model.state_dict(), model_path)

    for fname in gradient_update_files:
        os.remove(fname)

def avg_aggr_gradients(model, gradient_updates): 
    aggregated_gradients = gradient_updates[0]
    for gradient in gradient_updates[1:]:
        for name, param in model.named_parameters():
            aggregated_gradients[name] = torch.add(aggregated_gradients[name], gradient[name])

    for name, param in model.named_parameters():
        aggregated_gradients[name] /= len(gradient_updates)
        param.grad = aggregated_gradients[name]


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
    gradients = {name: p.grad.data for name, p in model.named_parameters()}
    fname = f"gradient_updates_{node_num}.pt"
    with open(fname, 'wb') as f:
            save(gradients, f)
    return fname

def wait_for_gradient_updates(model, node_num):
    fname = f"gradient_updates_{node_num}.pt"
    
    while not os.path.exists(output_model_file):
                time.sleep(1)

    with open(output_model_file, 'rb') as f:
            model.load_state_dict(load(f)) 
            os.remove(output_model_file)

def complete_training(model, node_num):
    fname = f"gradient_last_updates_{node_num}.pt"
    save(model.state_dict(), f"~/trained_{node_num}.pt")

    with open(fname, 'wb') as f:
            save(model.state_dict(), f)

    with open(f"training_complete_{node_num}", 'w'):
        pass