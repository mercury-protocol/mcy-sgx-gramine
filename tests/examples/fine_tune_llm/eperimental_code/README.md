LLM fine tuning

* TRL - Transformer reinforcement Learning
    * https://huggingface.co/docs/trl/index
* PEFT - Parameter-Efficient Fine-Tuning
    * https://huggingface.co/docs/peft/index

TRL has 3 steps:
* SFT - Supervised Fine-Tuning
    * adapt pre-trained transformer models, such as BERT or GPT, to specific tasks. During SFT, the pre-trained model is fine-tuned on a supervised dataset where both inputs and corresponding outputs are provided. The objective is to adjust the model's parameters to minimize a task-specific loss function, typically using gradient descent optimization. SFT is commonly used in scenarios where labeled data is available for the target task, such as text classification, named entity recognition, or sentiment analysis
* RM - Reward Modelling
    * learn a reward function from human demonstrations or expert knowledge. In RM, a separate reward model is trained to predict rewards based on observed state-action pairs. The learned reward function is then used to guide the agent's behaviour during training by providing feedback on the desirability of different actions. RM is useful in scenarios where designing a reward function directly is challenging or where the reward function needs to be learned from human preferences or domain expertise
* PPO - Proximal Policy Optimisation
    * train policies in environments modelled as Markov Decision Processes (MDPs). PPO aims to improve stability and sample efficiency by constraining policy updates to be close to the previous policy using a clipped objective function. The key idea behind PPO is to prevent large policy changes during training, thereby maintaining stability and improving convergence. PPO is widely used in RL scenarios, including robotics, game playing, and continuous control tasks, due to its simplicity, stability, and effectiveness

A good intro on how to fine-tune LLMs:  (it uses LoRA from PEFT)
* https://www.youtube.com/watch?v=eC6Hd1hFvos
    * https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs/fine-tuning

Trainer classes to implement the 3 steps above:
* IterativeSFTTrainer:
    * https://huggingface.co/docs/trl/trainer#trl.IterativeSFTTrainer
    * https://huggingface.co/docs/trl/en/iterative_sft_trainer
* RewardTrainer:  (this doesn’t have the step() method)
    * https://huggingface.co/docs/trl/trainer#trl.RewardTrainer
* PPOTrainer:
    * https://huggingface.co/docs/trl/trainer#trl.PPOTrainer

An example for a training loop:
* https://huggingface.co/docs/trl/ppo_trainer#starting-the-training-loop
* probably something similar will have to be implemented in Mercury for both SFT and PPO

TODOs:
* first implement just a normal case with no distributed training
* in all cases the model attribute of the trainer is either an nn.Module class or a child class of it so the saving of gradients should be the same as in the image processing example or very similar
    * thus the gradient aggregation should be also very similar
* currently in the leader we do optimizer.step(), but in the LLM case we should do trainer.step()
    * we have to figure out how to implement it properly
* since we are using trainers and models we have to figure out how to load them
