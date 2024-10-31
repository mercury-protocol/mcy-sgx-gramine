import shutil
import torch
from pathlib import Path
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer

from mcy_dist_ai.constants import BASE_DIR, GRADIENT_FILE, OUTPUT_DIR
from mcy_dist_ai.logger import logger
from mcy_dist_ai.utils import user_script


class VulkanCallback(TrainerCallback):
    # TODO: finish the implementation of this class
    #  it is used in the fully automated LLM fine tuning case, where ROLE == "WORKER-LLM"
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def save_gradients(self):
        gradient = {name: param.data for name, param in self.trainer.model.named_parameters() if param.requires_grad}
        torch.save(gradient, BASE_DIR / GRADIENT_FILE)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # self.trainer.model = load_model(path=STATE_DICT_PATH, delete_file=True)
        # optimizer = load_optimizer(model)
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.save_gradients()
        # checkpoint(epoch=state.epoch, batch_idx=batch_idx)

        # if state.global_step == state.max_steps:
        #     self.signal_worker_finished()
        # else:
        #     await self.wait_state_dict()
        print(state.epoch, state.global_step, state.max_steps)


class Worker:
    @staticmethod
    async def fine_tune_llm():
        # TODO: make this implementation compatible with the original code flow:
        #  - distributed training
        #  - use Mercury user script format
        #  - no separate role for llm training

        logger.info("Worker started - Fine tune LLM")

        trainer = user_script.create_trainer()
        callback = VulkanCallback(trainer=trainer)
        trainer.add_callback(callback)
        shutil.rmtree(trainer.args.output_dir)
        trainer.args.output_dir = OUTPUT_DIR / Path(trainer.args.output_dir).name

        trainer.train()
