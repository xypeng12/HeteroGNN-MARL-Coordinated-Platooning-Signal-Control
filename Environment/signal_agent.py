import traci
import numpy as np
from Environment.network_information import get_neighbor_signal, get_phase_set,get_controlled_lanes

class SignalAgent:
    def __init__(self, name,agent_name,control_interval):
        self.fingerprint = [] # local policy
        self.name = name
        self.agent_name = agent_name
        self.control_interval=control_interval
        self.coef_vehwait = 1 #0.2 converge to about 100
        self.relevant_sp = {}
        self.num_state = 0 # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        self.reward=0
        self.wait=0
        self.vehwait=0

        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1

        self.get_approach()
        self.get_relevant_signal()
        self.get_phaseset()
        self.get_obs_space()
        self.get_action_space()


    def get_action_space(self):
        self.action_space=len(self.phases)

    def get_obs_space(self):
        self.obs_space=len(self.ilds_in)

    def action(self,action):
        phase=self.get_phase(action)
        traci.trafficlight.setRedYellowGreenState(self.name, phase)
        traci.trafficlight.setPhaseDuration(self.name, self.control_interval)
    def get_phase(self,action):
        cur_phase = self.phases[action]
        return cur_phase

    def get_approach(self):
        self.ilds_in = get_controlled_lanes(self.name)

    def get_phaseset(self):
        self.phases=get_phase_set(self.name)#可选相位的集合

    def get_relevant_signal(self):
        self.relevant_ss=get_neighbor_signal(self.name)
    def get_obs(self):
        cur_state = []
        for ild in self.ilds_in:
            cur_wave = traci.lane.getLastStepHaltingNumber(ild)
            cur_state.append(cur_wave)
        return cur_state
    def get_reward(self):
        waits=[]
        vehwaits=[]

        for ild in self.ilds_in:
            waits.append(traci.lane.getLastStepHaltingNumber(ild)*self.control_interval)
            max_pos = 0
            veh_wait = 0
            if traci.lane.getLastStepHaltingNumber(ild) == 0:
                veh_wait = 0
            else:
                vehs = traci.lane.getLastStepVehicleIDs(ild)
                for vid in vehs:
                    pos = traci.vehicle.getLanePosition(vid)
                    if pos > max_pos:
                        max_pos = pos
                        veh_wait = traci.vehicle.getWaitingTime(vid)
            vehwaits.append(veh_wait)
        wait = np.sum(np.array(waits)) if len(waits) else 0
        vehwait = np.sum(np.array(vehwaits)) if len(vehwaits) else 0
        reward = -wait - self.coef_vehwait * vehwait
        return reward