import os
import time

from mcy_dist_ai import parse_worker_nodes_count, aggregate_gradients, aggregate_gradients_and_save_model


if __name__ == "__main__":
    worker_nodes_count = parse_worker_nodes_count()

    while not os.path.exists("training_complete"):
        files_in_current_directory = os.listdir()
        gradient_update_files = [file for file in files_in_current_directory if 'gradient_updates' in file]

        while len(gradient_update_files) != worker_nodes_count and not os.path.exists("training_complete"):
            files_in_current_directory = os.listdir()
            gradient_update_files = [file for file in files_in_current_directory if 'gradient_updates' in file]

            time.sleep(1)

        aggregate_gradients()
        time.sleep(1)

    # aggregate last updates
    files_in_current_directory = os.listdir()
    gradient_update_files_last = [file for file in files_in_current_directory if 'gradient_last_updates' in file]
    aggregate_gradients_and_save_model()

    
    