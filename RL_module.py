import copy

from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
from typing import (
    Any,
    Dict)
import gymnasium as gym

from ray.rllib.utils import override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.configs import (
    MLPEncoderConfig,
)
import torch
import torch.nn as nn

from Environment.network_information import sa_get_relevant_agent

relevant_ss_set,relevant_sp_num=sa_get_relevant_agent()

class SP_PPOCatalog(PPOCatalog):
    def _get_encoder_config(
            cls,
            observation_space: gym.Space,
            model_config_dict: dict,
            action_space: gym.Space = None,
            view_requirements=None,
    ) -> ModelConfig:
        encoder_config = MLPEncoderConfig(
            input_dims=[model_config_dict["contacter_dim"]],
            hidden_layer_dims=[model_config_dict["contacter_out"]],
            hidden_layer_activation="relu",
            hidden_layer_use_layernorm=False,
            output_layer_dim=None,  # maybe None or an int
        )
        return encoder_config

class SPTorchRLModuleWithSharedGlobalEncoder(PPOTorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self, config,shared_encoder: nn.Module,input_dim,self_hidden_dim
                ,module_id:None,scenario=None,signal_agent_list=None,pre_training=False
    ) -> None:
        super().__init__(config=config)
        self.shared_encoder = shared_encoder
        self.self_encoder = nn.Sequential(torch.nn.Linear(input_dim, self_hidden_dim))
        self.module_id=module_id
        self.signal_agent_list=signal_agent_list
        self.scenario=scenario
        self.pre_training=pre_training

    def _forward_inference(self, batch: NestedDict) -> Dict[str, Any]:
        batch_copy = copy.deepcopy(batch)
        batch_copy["obs"] = self.shared_obs_contact(batch_copy["obs"])
        return super()._forward_inference(batch_copy)

    @override(PPOTorchRLModule)
    def _forward_exploration(self, batch: NestedDict, **kwargs) -> Dict[str, Any]:
        batch_copy = copy.deepcopy(batch)
        batch_copy["obs"] = self.shared_obs_contact(batch_copy["obs"])
        return super()._forward_exploration(batch_copy)

    @override(PPOTorchRLModule)
    def _forward_train(self, batch: NestedDict) -> Dict[str, Any]:
        batch_copy = copy.deepcopy(batch)
        batch_copy["obs"] = self.shared_obs_contact(batch_copy["obs"])
        return super()._forward_train(batch_copy)

    @override(PPOTorchRLModule)
    def _compute_values(self, batch, device=None):
        batch_copy = copy.deepcopy(batch)
        batch_copy = convert_to_torch_tensor(batch_copy, device=device)
        batch_copy["obs"] = self.shared_obs_contact(batch_copy["obs"])
        return super()._compute_values(batch_copy)
    def shared_obs_contact(self, obs):

        enc = self.self_encoder(obs['local'])
        contact = [enc]

        if self.scenario == "signal_platoon" or self.pre_training:
            if self.module_id.startswith("signal"):
                # signal agent
                for i in range(relevant_sp_num):
                    relevant_PA_obs=obs["platoon%d" % i]
                    relevant_PA_enc=[]
                    for b in range(len(relevant_PA_obs)):
                        if relevant_PA_obs[b]==[-1,-1,-1,-1,-1]:
                            encoded_value= torch.full((self.config.model_config_dict["sp_hidden_dim"],), -1.0)
                        else:
                            encoded_value=self.shared_encoder['sp_platoon'](relevant_PA_obs[b])
                        relevant_PA_enc.append(encoded_value)
                    relevant_PA_enc=torch.stack(relevant_PA_enc)
                    contact.append(relevant_PA_enc)
                for relevant_SA_id in relevant_ss_set[self.module_id]:
                    relevant_SA_enc = self.shared_encoder['ss_'+relevant_SA_id](obs[relevant_SA_id])
                    contact.append(relevant_SA_enc)
            else:
                # platoon agent
                signal_ids = [self.signal_agent_list[int(idx)] if idx != -1 else -1 for idx in obs['signal_id_index']]
                relevant_SA_encs=[]
                for b,sa in enumerate(signal_ids):
                    if sa==-1:
                        encoded_value= torch.full((self.config.model_config_dict["ps_hidden_dim"],), -1.0)
                    else:
                        encoded_value=self.shared_encoder['ps_'+sa](obs['signal'][b])
                    relevant_SA_encs.append(encoded_value)

                relevant_SA_enc = torch.stack(relevant_SA_encs)  # This assumes each encoder output is a tensor
                contact.append(relevant_SA_enc)

        else:
            if self.module_id.startswith("signal"):
                # signal agent
                for relevant_SA_id in relevant_ss_set[self.module_id]:
                    relevant_SA_enc = self.shared_encoder['ss_'+relevant_SA_id](obs[relevant_SA_id])
                    contact.append(relevant_SA_enc)

        cat = torch.cat(contact, dim=-1)
        return cat



class SPTorchMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):
    def __init__(self, config:MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    @override(MultiAgentRLModule)
    def setup(self):
        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        signal_agent_list=module_spec.model_config_dict["signal_agent_list"]

        input_dim_dict=module_spec.model_config_dict["input_dim_dict"]
        shared_input_dim_dict=module_spec.model_config_dict["shared_input_dim_dict"]
        shared_output_dim_dict=module_spec.model_config_dict["shared_output_dim_dict"]


        scenario=module_spec.model_config_dict["scenario"]
        pre_training=module_spec.model_config_dict["pre_training"]
        shared_encoder=nn.ModuleDict()
        for id, input_dim in shared_input_dim_dict.items():
            shared_encoder[id] = nn.Sequential(
                torch.nn.Linear(input_dim, shared_output_dim_dict[id]))

        rl_modules = {}

        for module_id, module_spec in self.config.modules.items():
            rl_modules[module_id] = SPTorchRLModuleWithSharedGlobalEncoder(
                config=module_spec.get_rl_module_config(),
                shared_encoder=shared_encoder,
                input_dim=input_dim_dict[module_id],
                self_hidden_dim=module_spec.model_config_dict["self_hidden_dim"],
                module_id=module_id,
                scenario=scenario,
                signal_agent_list=signal_agent_list,
                pre_training=pre_training
            )

        self._rl_modules = rl_modules

        framework = None
        for module_id,module_spec in self.config.modules.items():
            if framework is None:
                framework = self._rl_modules[module_id].framework
            else:
                assert self._rl_modules[module_id].framework in [None, framework]
        self.framework = framework