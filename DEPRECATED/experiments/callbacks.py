import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=1, reward_setting=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self):
        super()._on_step()
        try:
            training_env = self.model.get_env().envs[0].env.unwrapped.gym_env
        except AttributeError:
            training_env = self.model.get_env().get_attr('unwrapped')[0]
            # training_env = self.model.get_env().envs[0].env.unwrapped
        # action = training_env.last_action_taken

        log_dict = {}
        obs = training_env.current_state_repr
        if len(obs.shape) == 2:
            obs = obs[-1]
        observation = obs.reshape((-1,) + obs.shape)
        observation = obs_as_tensor(observation, self.model.device)
        log_dict.update({f"observations/{i}": observation[0][i].item() for i in range(len(observation))})

        log_dict.update({
            # "train/last_action": action,
            # "train/average_service_time": training_env.core_env.state.trackers.average_service_time,
        })
        wandb.log(log_dict, step=self.num_timesteps)

        # if isinstance(self.model, DQN) or isinstance(self.model, SAC):
        #     if self.model.num_timesteps % self.log_interval == 0:
        #         self._dump_logs_to_wandb()

        return True

    def _dump_logs_to_wandb(self):
        for key, value in self.model.logger.name_to_value.items():
            wandb.log({f"train/{key}": value}, step=self.num_timesteps)
