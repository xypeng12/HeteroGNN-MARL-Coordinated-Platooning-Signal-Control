import ray
from ray.rllib.algorithms.ppo import PPOConfig
from Environment.multiagent_environment import SignalPlatoonEnv
from ray import tune
import os
import sys
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from RL_module import SPTorchMultiAgentModuleWithSharedEncoder, SPTorchRLModuleWithSharedGlobalEncoder, SP_PPOCatalog
from ray.rllib.algorithms import Algorithm

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
from Environment.network_information import sa_get_relevant_agent

relevant_ss_set, relevant_sp_num = sa_get_relevant_agent()
ray.init()

netfile = 'Environment/network_2_3/network23.net.xml'
routefile = 'Environment/network_2_3/network23.rou.xml'
cfg_file = 'Environment/network_2_3/network23.sumocfg'
add_file='Environment/network_2_3/network23.add.xml'
config = (PPOConfig().experimental(_disable_preprocessor_api=True).api_stack(
    enable_rl_module_and_learner=True,
    enable_env_runner_and_connector_v2=True,
).env_runners(num_env_runners=4)
          ).training(
    model={
        "_disable_preprocessor_api": True,
        "use_new_env_runners": True,
    },
    train_batch_size=1000,
    use_kl_loss=False,
)
config.sample_timeout_s = 60
config.rollout_fragment_length = 50

def generate_module_specs(env, scenario, pre_training, signal_agent_list):
    #policy->module
    module_spec_dict = {}

    input_dim_dict = env.input_dim_dict
    shared_input_dim_dict = env.shared_input_dim_dict
    shared_output_dim_dict={}
    shared_hidden_dim = {'ss': 8, 'sp':8, 'ps': 32}  # 8
    self_hidden_dim = {'signal': 16, 'platoon': 32}
    self_hidden_dim_set={}
    contacter_dim_set = {}
    contacter_out = 32

    for id in shared_input_dim_dict.keys():
        if id.startswith('ss'):
            shared_output_dim_dict[id] = shared_hidden_dim['ss']
        elif id.startswith('sp'):
            shared_output_dim_dict[id] = shared_hidden_dim['sp']
        elif id.startswith('ps'):
            shared_output_dim_dict[id] = shared_hidden_dim['ps']

    for policy in env.observation_space_policy:
        if policy.startswith("signal"):
            self_hidden_dim_set[policy]=self_hidden_dim['signal']
        else:
            self_hidden_dim_set[policy]=self_hidden_dim['platoon']

        if scenario == "signal_platoon" or pre_training:
            if policy.startswith("signal"):
                contacter_dim_set[policy]  = len(relevant_ss_set[policy])*shared_hidden_dim['ss'] + relevant_sp_num*shared_hidden_dim['sp']+self_hidden_dim_set[policy]
            else:
                contacter_dim_set[policy]  = shared_hidden_dim['ps']+self_hidden_dim_set[policy]
        else:
            if policy.startswith("signal"):
                contacter_dim_set[policy]  = len(relevant_ss_set[policy])*shared_hidden_dim['ss']+self_hidden_dim_set[policy]
            else:
                contacter_dim_set[policy]  = self_hidden_dim_set[policy]


    for policy in env.observation_space_policy:
        module_spec_dict[policy] = SingleAgentRLModuleSpec(
            module_class=SPTorchRLModuleWithSharedGlobalEncoder,
            observation_space=env.observation_space_policy[policy], action_space=env.action_space_policy[policy],
            model_config_dict={"input_dim_dict": input_dim_dict,
                               'self_hidden_dim': self_hidden_dim_set[policy],
                               "shared_input_dim_dict": shared_input_dim_dict,
                               "shared_output_dim_dict": shared_output_dim_dict,
                               "contacter_dim": contacter_dim_set[policy],
                               "contacter_out": contacter_out,
                               "scenario": scenario, "pre_training": pre_training,
                               "signal_agent_list": signal_agent_list,
                               "sp_hidden_dim":shared_hidden_dim['sp'],
                               "ps_hidden_dim":shared_hidden_dim['ps']
                               },
            catalog_class=SP_PPOCatalog)
    return module_spec_dict


def evaluate_policy(config,scenario,output_dir,routefile=routefile,check_point_dir=None):
    pre_training = False
    env = SignalPlatoonEnv(scenario=scenario,
                           net_file=netfile,
                           route_file=routefile,
                           add_file=add_file,
                           cfg_file=cfg_file,
                           output_dir=output_dir,
                           pre_training=pre_training,
                           use_gui=False,
                           direct_start=True
                           )
    tune.register_env("env", lambda _: env)

    policy_set, policy_map, signal_policies = env.get_policy_dict()
    signal_agent_list = env.signal_agent_ids
    policy_map_fn = lambda agent_id, *args, **kwargs: policy_map[agent_id]
    module_specs = generate_module_specs(env, scenario, pre_training, signal_agent_list)

    for policy_id in list(policy_set):
        path = os.path.join(check_point_dir, "learner", "module_state", policy_id)
        module_spec = module_specs[policy_id]
        module_spec.load_state_path = path
        module_specs[policy_id] = module_spec

    config_train = config.environment("env").multi_agent(policies=policy_set,
                                                                       policy_mapping_fn=policy_map_fn,
                                                                       policies_to_train=list(policy_set)).rl_module(
        model_config_dict={"vf_share_layers": True}, rl_module_spec=MultiAgentRLModuleSpec(
            marl_module_class=SPTorchMultiAgentModuleWithSharedEncoder,
            module_specs=module_specs))
    train = config_train.build()
    train.train()

def evaluate_diff_demand(config,scenario,check_point_dir):
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

    for ratio in ratios:
        file_name = f'network23_demand-rate{ratio}.rou.xml'
        demand_route_file = f'Environment/network_2_3/{file_name}'
        output_dir=f'results/diff_demand/output{ratio}.csv'
        evaluate_policy(config, scenario,output_dir, routefile=demand_route_file, check_point_dir=check_point_dir)


def evaluate_diff_CAV_rate(config,scenario,check_point_dir):
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 遍历比率列表并生成文件名
    for ratio in ratios:
        file_name = f'network23_CAV-rate{ratio}.rou.xml'
        demand_route_file = f'Environment/network_2_3/{file_name}'
        output_dir=f'results/diff_CAV_rate/output{ratio}.csv'
        evaluate_policy(config, scenario, output_dir, routefile=demand_route_file, check_point_dir=check_point_dir)

def write_to_file(message):
    with open('results/output.txt', 'a', encoding='utf-8') as file:
        file.write("\n"+message + "\n")
        file.flush()  # 确保立即写入到文件

def joint_train_and_evaluate():
    output_dir=f'results/joint_output.csv'
    scenario = 'signal_platoon'
    pre_training = False
    env = SignalPlatoonEnv(scenario=scenario,
                           net_file=netfile,
                           route_file=routefile,
                           add_file=add_file,
                           cfg_file=cfg_file,
                           output_dir=output_dir,
                           pre_training=pre_training,
                           use_gui=False,
                           direct_start=True,
                           )
    tune.register_env("env", lambda _: env)
    final_checkpoint='results/model/tmpi42wgtxb'

    write_to_file('evaluate_under_diff_CAV_rate')
    evaluate_diff_CAV_rate(config,'signal_platoon', final_checkpoint)


def signal_train_and_evaluate():
    #only exist signal
    output_dir = f'results/signal_output.csv'
    scenario = 'signal'
    pre_training = False
    env = SignalPlatoonEnv(scenario=scenario,
                           net_file=netfile,
                           route_file=routefile,
                           add_file=add_file,
                           cfg_file=cfg_file,
                           output_dir=output_dir,
                           pre_training=pre_training,
                           use_gui=False,
                           direct_start=True,
                           )
    tune.register_env("env", lambda _: env)
    policy_set, policy_map, signal_policies = env.get_policy_dict()
    signal_agent_list = env.signal_agent_ids
    policy_map_fn = lambda agent_id, *args, **kwargs: policy_map[agent_id]
    module_specs = generate_module_specs(env, scenario, pre_training, signal_agent_list)
    for iteration in range(4):
        if iteration == 0:
            write_to_file('train_signal' + str(iteration))
            config_signal_first = config.environment("env").multi_agent(policies=policy_set,
                                                                        policy_mapping_fn=policy_map_fn,
                                                                        policies_to_train=signal_policies).rl_module(
                model_config_dict={"vf_share_layers": True}, rl_module_spec=MultiAgentRLModuleSpec(
                    marl_module_class=SPTorchMultiAgentModuleWithSharedEncoder,
                    module_specs=module_specs))

            train_signal = config_signal_first.build()

            for i in range(200):  # 50
                # iteration
                train_signal.train()
            check_point_dir = train_signal.save().checkpoint.path
            write_to_file('check_point_dir:' + check_point_dir)
        else:
            write_to_file('train_signal' + str(iteration))
            trainer_signal = Algorithm.from_checkpoint(checkpoint=check_point_dir, policy_ids=policy_set,
                                                       policy_mapping_fn=policy_map_fn,
                                                       policies_to_train=signal_policies)
            for i in range(200):  # 100
                trainer_signal.train()

            check_point_dir = trainer_signal.save().checkpoint.path
            trainer_signal.stop()
            write_to_file('check_point_dir:' + check_point_dir)

    write_to_file('evaluate_signal_train_policy:' + check_point_dir)
    output_dir = f'results/signal_eva_output.csv'
    evaluate_policy(config,'signal',output_dir=output_dir,check_point_dir=check_point_dir)

def platoon_train_and_evaluate():
    # only exist platoon
    output_dir=f'results/platoon_output.csv'
    scenario = 'platoon'
    pre_training = False
    env = SignalPlatoonEnv(scenario=scenario,
                           net_file=netfile,
                           route_file=routefile,
                           add_file=add_file,
                           cfg_file=cfg_file,
                           output_dir=output_dir,
                           pre_training=pre_training,
                           use_gui=False,
                           direct_start=True,
                           )
    tune.register_env("env", lambda _: env)
    policy_set, policy_map, signal_policies = env.get_policy_dict()
    signal_agent_list = env.signal_agent_ids
    policy_map_fn = lambda agent_id, *args, **kwargs: policy_map[agent_id]
    module_specs = generate_module_specs(env, scenario, pre_training, signal_agent_list)
    check_point_dir = None
    for iteration in range(4):
        if iteration == 0:
            write_to_file('train_platoon' + str(iteration))
            config_platoon_first = config.environment("env").multi_agent(policies=policy_set,
                                                                        policy_mapping_fn=policy_map_fn,
                                                                        policies_to_train=['platoon']).rl_module(
                model_config_dict={"vf_share_layers": True}, rl_module_spec=MultiAgentRLModuleSpec(
                    marl_module_class=SPTorchMultiAgentModuleWithSharedEncoder,
                    module_specs=module_specs))

            train_platoon = config_platoon_first.build()

            for i in range(200):  # 50
                # iteration
                train_platoon.train()
            check_point_dir = train_platoon.save().checkpoint.path
            write_to_file('check_point_dir:' + check_point_dir)
        else:
            write_to_file('train_platoon' + str(iteration))
            train_platoon = Algorithm.from_checkpoint(checkpoint=check_point_dir, policy_ids=policy_set,
                                                       policy_mapping_fn=policy_map_fn,
                                                       policies_to_train=['platoon'])
            for i in range(200):  # 100
                train_platoon.train()

            check_point_dir = train_platoon.save().checkpoint.path
            train_platoon.stop()
            write_to_file('check_point_dir:' + check_point_dir)

    write_to_file('evaluate_platoon_train_policy:'+check_point_dir)
    output_dir=f'results/platoon_eva_output.csv'
    evaluate_policy(config,'platoon',output_dir=output_dir,check_point_dir=check_point_dir)


joint_train_and_evaluate()
