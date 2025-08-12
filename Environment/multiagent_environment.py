import os
import sys
import numpy as np
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary # noqa
import traci # noqa
from Environment.signal_agent import SignalAgent
from Environment.platoon_agent import Platoon_System
import gymnasium as gym
import sumolib
from ray.rllib.env import MultiAgentEnv
import csv
from Environment.network_information import sa_get_relevant_agent,read_signal_id


relevant_ss_set,relevant_sp_num=sa_get_relevant_agent()

class SignalPlatoonEnv(MultiAgentEnv):
    def __init__(self,scenario,net_file, route_file, add_file,cfg_file,output_dir, pre_training=False, use_gui=True, direct_start=True,random_signal=False,random_platoon=False):
        self.scenario =scenario # signal_platoon, signal, platoon
        self.output_dir=output_dir
        self.pre_training=pre_training
        self._net= net_file
        self._route= route_file
        self._add=add_file
        self._cfg= cfg_file
        self.use_gui= use_gui
        self.direct_start = direct_start
        self.random_signal=random_signal
        self.random_platoon=random_platoon
        self.current_episode = 0
        self.control_interval = 5

        self.warmup_time = 200
        self.cur_sec = self.warmup_time
        self.temp =0
        self.episode_length_sec = 600+self.warmup_time


        self.obs_signal_size = 6 #12
        self.ac_signal_size = 3
        self.nearest_platoon_number = 3

        self.obs_platoon_size =5
        self.max_platoon_size = 10  # max platoon size=10

        self.ac_platooon_size = 5
        self.PA_num = 100

        self.distance_threshold=2
        self.timeheadway_threshold=2
        self.ave_signal_agent_reward = []
        self.ave_platoon_agent_reward = []

        self.platoon_system=Platoon_System(self.PA_num,self.max_platoon_size,self.distance_threshold,self.timeheadway_threshold,self.control_interval)

        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self._agent_ids=self.init_agents()

        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        self.observation_space, self.action_space=self.init_space()
        self.input_dim_dict,self.shared_input_dim_dict=self.init_input_dim_dict()

        self.terminateds = set()
        self.truncateds = set()
        super().__init__()

    def reset(self, *,seed=None, options=None):
        if self.temp==1:
            self.cur_sec=self.warmup_time
            traci.close()
        self.temp=1
        if self.scenario=='signal_platoon' or self.scenario=='platoon' or self.pre_training:
            self.platoon_system.reset()

        self.current_episode += 1

        #reset sumo
        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '--additional-files',self._add,
                    '--quit-on-end',
                    "--collision.action", "none",
                    '--waiting-time-memory', '10000',
                    '--random',
                    '--no-step-log',
                    '--no-warnings']

        traci.start(sumo_cmd)
        traci.simulation.step(time=self.warmup_time)

        self.ave_signal_agent_reward = []
        self.ave_platoon_agent_reward = []
        self.reset_info()
        obs_platoon = {}
        obs_signal = {}

        if self.scenario=='signal_platoon' or self.scenario=='platoon':
            obs_platoon_each={"local":np.array([-1]*self.obs_platoon_size, dtype=np.float32)}
            if self.scenario=="signal_platoon" or self.pre_training:
                obs_platoon_each["signal"] = np.array([-1]*self.obs_signal_size, dtype=np.float32)
                obs_platoon_each["signal_id_index"]=np.array([-1], dtype=np.float32)
            for agent_id in self.platoon_agent_ids:
                obs_platoon[agent_id] = obs_platoon_each

        if self.scenario=='signal_platoon' or self.scenario=='signal':
            for agent_id in self.signal_agent_ids:
                obs_signal[agent_id]={"local":np.array([0]*self.obs_signal_size, dtype=np.float32)}
                for relevant_SA_id in relevant_ss_set[agent_id]:
                    obs_signal[agent_id][relevant_SA_id] = np.array([0]*self.obs_signal_size, dtype=np.float32)
                if self.scenario == "signal_platoon" or self.pre_training:
                    for i in range(self.nearest_platoon_number):
                        obs_signal[agent_id]["platoon%d"%i]=np.array([-1]*self.obs_platoon_size, dtype=np.float32)

        if self.scenario == 'signal_platoon':
            obs= {**obs_signal, **obs_platoon}
        elif self.scenario=='signal':
            obs={**obs_signal}
        else:
            obs={**obs_platoon}


        info = {agent_id: {} for agent_id in obs.keys()}

        return obs,info

    def init_agents(self):
        self.signal_agent_ids = read_signal_id()
        if self.scenario=='signal_platoon' or self.scenario=='signal' or self.pre_training:
            signals = {}
            for signal_agent_id in self.signal_agent_ids:
                id=signal_agent_id[6:]
                signals[signal_agent_id] = SignalAgent(id,signal_agent_id,self.control_interval)
            self.signals = signals
        if self.scenario=='signal_platoon' or self.scenario=='platoon' or self.pre_training:
            self.platoon_agent_ids=self.platoon_system.platoon_id_list
        if self.scenario=='signal_platoon':
            return set(self.signal_agent_ids+self.platoon_agent_ids)
        elif self.scenario=='signal':
            return set(self.signal_agent_ids)
        elif self.scenario=='platoon':
            return set(self.platoon_agent_ids)
    def get_policy_dict(self):
        policy_map={}
        policy_set=set()
        if self.scenario == 'signal_platoon' or self.scenario == 'signal':
            # 循环遍历signal列表，创建Policy Spec并存储到字典中
            for agent_id in self.signal_agent_ids:
                policy_set.add(agent_id)
                policy_map[agent_id]=agent_id

        if self.scenario == 'signal_platoon' or self.scenario == 'platoon':
            policy_set.add("platoon")
            for agent_id in self.platoon_agent_ids:
                policy_map[agent_id]="platoon"

        signal_policies=[]
        for agent_id in self.signal_agent_ids:
            signal_policies.append(agent_id)
        return policy_set,policy_map,signal_policies
    def init_input_dim_dict(self):
        shared_input_dim_dict={}
        input_dim_dict={}
        if self.pre_training or self.scenario=='signal_platoon':
            for agent_id in self.signal_agent_ids:
                shared_input_dim_dict['ss_' + agent_id] = self.obs_signal_size
                shared_input_dim_dict['ps_' + agent_id] = self.obs_signal_size
                input_dim_dict[agent_id]=self.obs_signal_size

            shared_input_dim_dict['sp_platoon']=self.obs_platoon_size
            input_dim_dict['platoon']=self.obs_platoon_size

            return input_dim_dict,shared_input_dim_dict

        if self.scenario == 'platoon':
            input_dim_dict['platoon']=self.obs_platoon_size
            return input_dim_dict,shared_input_dim_dict

        if self.scenario == 'signal':
            for agent_id in self.signal_agent_ids:
                shared_input_dim_dict['ss_' + agent_id] = self.obs_signal_size
                input_dim_dict[agent_id]=self.obs_signal_size

            return input_dim_dict,shared_input_dim_dict

    def init_space(self):
        observation_space_set = {}
        action_space_set = {}

        observation_space_policy={}
        action_space_policy={}
        obs_platoon_min=[-1.,-1.,-1.,-1.,-1.]
        obs_platoon_max=[500.,10.,20.,5.,5.]

        if self.scenario == 'signal_platoon' or self.scenario == 'signal':
            # 循环遍历signal列表，创建Policy Spec并存储到字典中
            for agent_id in self.signal_agent_ids:
                obs_space= {"local":gym.spaces.Box(np.float32(-1), np.float32(100), (self.obs_signal_size,))}
                for relevant_SA_id in relevant_ss_set[agent_id]:
                    obs_space[relevant_SA_id] = gym.spaces.Box(np.float32(-1), np.float32(100),
                                                                       (self.obs_signal_size,))
                if self.scenario == 'signal_platoon' or self.pre_training:
                    for i in range(self.nearest_platoon_number):
                        obs_space["platoon%d"%i]=gym.spaces.Box(np.array(obs_platoon_min),np.array(obs_platoon_max),dtype=np.float32)

                obs_space = gym.spaces.Dict(obs_space)
                observation_space_set[agent_id] = obs_space

                action_space = gym.spaces.Discrete(self.ac_signal_size)
                action_space_set[agent_id] = action_space

                observation_space_policy[agent_id] = obs_space
                action_space_policy[agent_id] = action_space

        if self.scenario == 'signal_platoon' or self.scenario == 'platoon':
            obs_space = {"local": gym.spaces.Box(np.array(obs_platoon_min),np.array(obs_platoon_max),dtype=np.float32)}
            if self.scenario == 'signal_platoon' or self.pre_training:
                obs_space["signal"] = gym.spaces.Box(np.float32(-1), np.float32(100), (self.obs_signal_size,))
                obs_space["signal_id_index"] = gym.spaces.Box(np.float32(-1), np.float32(100), (1,))

            obs_space = gym.spaces.Dict(obs_space)
            action_space = gym.spaces.Discrete(self.ac_platooon_size)

            for agent_id in self.platoon_agent_ids:
                observation_space_set[agent_id] = obs_space
                action_space_set[agent_id] = action_space

            observation_space_policy['platoon'] = obs_space
            action_space_policy['platoon'] = action_space

        observation_space = gym.spaces.Dict(observation_space_set)
        action_space = gym.spaces.Dict(action_space_set)

        self.observation_space_policy=gym.spaces.Dict(observation_space_policy)
        self.action_space_policy=gym.spaces.Dict(action_space_policy)

        return observation_space, action_space


    def get_agent_ids(self):
        return self._agent_ids
    @property

    def seed(self,seed=None):
        return seed

    def simulate(self):
        self.avg_speed_eachsecond=[]
        self.fuel_consumption_eachcontrolstep=0
        for _ in range(self.control_interval):
            current_vehicles=traci.vehicle.getIDList()
            if self.scenario=='signal_platoon' or self.scenario=='platoon' or self.random_platoon:
                for agent_id in self.platoon_system.platoon_set:
                    self.platoon_system.platoon_set[agent_id].update_reward_info(current_vehicles)
                    self.platoon_system.platoon_set[agent_id].do_action(current_vehicles)

            traci.simulation.step()
            self.update_step_info()

        self.avg_speed_eachcontrolstep=np.mean(self.avg_speed_eachsecond)
        self.cur_sec += self.control_interval

    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            if agent_id.startswith("signal"):
                signal_agent=self.signals[agent_id]
                signal_agent.action(action)
            if agent_id.startswith("platoon"):
                if agent_id in self.platoon_system.platoon_set:
                    self.platoon_system.platoon_set[agent_id].save_action(action)

        if self.scenario=='signal' and self.random_platoon:
            for agent_id in self.platoon_system.platoon_set:
                random_action = np.random.randint(0, self.ac_platooon_size)
                self.platoon_system.platoon_set[agent_id].save_action(random_action)

        if self.scenario=='platoon' and self.random_signal:
            for tls in traci.trafficlight.getIDList():
                random_phase_index = np.random.randint(0, self.ac_signal_size)
                traci.trafficlight.setPhase(tls,random_phase_index)
                traci.trafficlight.setPhaseDuration(tls, self.control_interval)

        self.simulate()

        if self.scenario=='signal_platoon' or self.scenario=='platoon' or self.pre_training:
            self.platoon_system.update()

        done0 = False
        if self.cur_sec >= self.episode_length_sec:
            done0 = True

        obs_platoon_local={}
        obs_signal_local={}
        obs_platoon={}
        obs_signal={}
        reward_platoon={}
        reward_signal={}

        relevant_ps_set = {}
        signal_agent_reward=[]
        platoon_agent_reward=[]
        if self.scenario=='signal_platoon' or self.scenario=='signal' or self.pre_training:
            for agent_id in self.signal_agent_ids:
                signal_agent=self.signals[agent_id]
                obs_signal_local[agent_id]=np.array(signal_agent.get_obs(), dtype=np.float32)
                reward_signal[agent_id]=signal_agent.get_reward()
                signal_agent_reward.append(reward_signal[agent_id])

        if self.scenario=='signal_platoon' or self.scenario=='platoon' or self.pre_training:

            for agent_id in self.platoon_system.platoon_set:
                platoon_agent=self.platoon_system.platoon_set[agent_id]
                obs_platoon_local[agent_id]=np.array(platoon_agent.get_obs(), dtype=np.float32)
                obs_platoon[agent_id] = {"local": obs_platoon_local[agent_id]}
                reward_platoon[agent_id]=platoon_agent.get_reward()
                platoon_agent_reward.append(reward_platoon[agent_id])

                if self.scenario=='signal_platoon' or self.pre_training:
                    ahead_signal, distance, _ = platoon_agent.get_ahead_signal()

                    if ahead_signal is not None:
                        relevant_sa = 'signal' + ahead_signal
                        relevant_ps_set[agent_id] = [relevant_sa, distance]
                        obs_index = [self.signal_agent_ids.index(relevant_sa)]
                        relevant_signal_obs = obs_signal_local[relevant_sa]
                    else:
                        obs_index=[-1]
                        relevant_signal_obs = np.array([-1] * self.obs_signal_size, dtype=np.float32)

                    obs_platoon[agent_id]["signal"] = relevant_signal_obs
                    obs_platoon[agent_id]["signal_id_index"]=np.array(obs_index, dtype=np.float32)

        if self.scenario=='signal_platoon' or self.scenario=='signal':
            for agent_id in self.signal_agent_ids:
                obs_signal[agent_id] = {"local": obs_signal_local[agent_id]}
                for relevant_SA_id in relevant_ss_set[agent_id]:
                    obs_signal[agent_id][relevant_SA_id] = obs_signal_local[relevant_SA_id]
                if self.scenario =='signal_platoon' or self.pre_training:
                    relevant_sp_set = reverse_relevant(relevant_ps_set, self.nearest_platoon_number,
                                                       self.signal_agent_ids)

                    for i in range(self.nearest_platoon_number):
                        relevant_PA_id = relevant_sp_set[agent_id][i]
                        if relevant_PA_id==-1:
                            obs_signal[agent_id]["platoon%d" % i] = np.array([-1]*self.obs_platoon_size, dtype=np.float32)
                        else:
                            obs_signal[agent_id]["platoon%d" % i] = obs_platoon_local[relevant_PA_id]

        self.ave_signal_agent_reward.append(np.mean(signal_agent_reward))

        if platoon_agent_reward !=[]:
            self.ave_platoon_agent_reward.append(np.mean(platoon_agent_reward))
        else:
            self.ave_platoon_agent_reward.append(0)

        if self.scenario == "signal_platoon":
            obs= {**obs_signal, **obs_platoon}
            reward={**reward_signal,**reward_platoon}
        elif self.scenario=="signal":
            obs={**obs_signal}
            reward = {**reward_signal}
        else:
            obs={**obs_platoon}
            reward = {**reward_platoon}

        info = {}
        terminateds = {}
        truncated={}
        for agent_id in obs.keys():
            terminateds[agent_id] = done0
            truncated[agent_id] = False
            info[agent_id] = {}

        if self.cur_sec==795:
            print(234,len(self.platoon_system.platoon_vehicle_dict),self.cur_sec)

        if self.scenario=='signal_platoon' or self.scenario=='platoon':
            for agent_id in self.platoon_system.platoon_set:
                platoon_agent = self.platoon_system.platoon_set[agent_id]
                if platoon_agent.condition==0:
                    terminateds[agent_id]=True
                    truncated[agent_id]=True

        terminateds["__all__"]= done0
        truncated["__all__"]= False

        if done0:
            vehicle_num= len(self.vehicle_for_episode)
            vehicle_output=vehicle_num-len(traci.vehicle.getIDList())

            speed=np.mean(self.avg_speed)
            metrics_info = {'avg_wait_time':  self.waiting_time/vehicle_num,'avg_fuel_consumption': self.fuel_consumption/vehicle_num,'ave_travel_time':self.travel_time/vehicle_num,
                            'veh*distance/fuelconsumption':self.veh_m/self.fuel_consumption,'veh*distance':self.veh_m,
                            'vehicle_num':vehicle_num,'output':vehicle_output,'avg_speed':speed,"episode":self.current_episode}
            if self.scenario=='signal_platoon' or self.scenario=="platoon":
                metrics_info['ave_reward_platoon']=np.mean(self.ave_platoon_agent_reward)
            if self.scenario == 'signal_platoon' or self.scenario == "signal":
                metrics_info['ave_reward_signal'] = np.mean(self.ave_signal_agent_reward)
            self.write_to_csv([metrics_info], self.output_dir)

        return obs,reward,terminateds,truncated,info


    def write_to_csv(self, data, filename):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def reset_info(self):
        self.edge_set = traci.edge.getIDList()
        self.fuel_consumption = 0
        self.avg_speed=[]
        self.waiting_time = 0
        self.travel_time=0
        self.veh_m=0
        self.vehicle_for_episode = set()


    def update_step_info(self):
        vehicles=traci.vehicle.getIDList()
        self.vehicle_for_episode.update(vehicles)

        fuel_consumption = 0
        waiting_time = 0
        veh_speed=0
        veh_sum=0
        travel_time=0
        for edge in self.edge_set:
            fuel_consumption +=traci.edge.getFuelConsumption(edge)
            waiting_time +=traci.edge.getLastStepHaltingNumber(edge)
            travel_time+=traci.edge.getLastStepVehicleNumber(edge)
            veh_num=traci.edge.getLastStepVehicleNumber(edge)
            speed=traci.edge.getLastStepMeanSpeed(edge)
            veh_speed+=veh_num*speed
            veh_sum+=veh_num

        self.avg_speed_eachsecond.append(veh_speed/veh_sum)
        self.avg_speed.append(veh_speed/veh_sum)
        self.fuel_consumption +=fuel_consumption
        self.travel_time+=travel_time
        self.veh_m+=veh_speed
        self.waiting_time += waiting_time

        self.fuel_consumption_eachcontrolstep += fuel_consumption




def reverse_relevant(input_dict, nearest_platoon_number, signal_set):
    if input_dict == {}:
        return {signal: [-1] * nearest_platoon_number for signal in signal_set}

    output_dict = {}
    for signal in signal_set:
        output_dict[signal] = [-1] * nearest_platoon_number

    for key, value in input_dict.items():
        s = value[0]
        if s in output_dict:
            signal_list = output_dict[s]
            signal_list.append(key)
    for signal, p_list in output_dict.items():
        if len(p_list) > nearest_platoon_number and -1 in p_list:
            p_list = [p for p in p_list if p != -1]
        valid_keys = sorted(p_list, key=lambda x: input_dict[x][1] if x != -1 else float('inf'))[:nearest_platoon_number]
        output_dict[signal] = valid_keys + [-1] * (nearest_platoon_number - len(valid_keys))

    return output_dict