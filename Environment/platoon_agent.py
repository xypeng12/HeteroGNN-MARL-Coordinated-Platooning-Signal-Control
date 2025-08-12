import traci
import numpy as np
from Environment.network_information import get_lane_information,phase_list

class PlatoonAgent:
    def __init__(self,agent_name,leading_vehicle,vehicles,max_platoon_size,distance_threshold,timeheadway_threshold,lane_information,control_interval):
        self.agent_name=agent_name
        self.vehicles=vehicles
        self.leading_vehicle = leading_vehicle
        self.max_platoon_size = max_platoon_size
        self.distance_threshold = distance_threshold
        self.timeheadway_threshold = timeheadway_threshold
        self.lane_information = lane_information
        self.control_interval = control_interval
        self.max_speed = 15.7
        self.action = None
        self.fuel_consumption = 0
        self.reward_speed = []
        self.condition = 1

    def get_following_path(self,vehicle):
        route = traci.vehicle.getRoute(vehicle)
        cur_lane = traci.vehicle.getLaneID(vehicle)
        cur_edge = traci.lane.getEdgeID(cur_lane)

        for i in range(len(route)):
            if route[i]==cur_edge or (cur_edge[0] == ':' and route[i].split('_')[1]==cur_edge.split('_')[0][1:]):
                if i+2<=len(route)-1:
                    return [route[i],route[i+1],route[i+2]]
                elif i+1<=len(route)-1:
                    return [route[i],route[i+1],-1]
                else:
                    return [route[i],-1,-1]


    def update_vehicles(self):
        # let leading vehicle become milestone to find ahead vehicles and following vehicles

        vehicles=[self.leading_vehicle]

        following_path=self.get_following_path(self.leading_vehicle)

        leader=self.leading_vehicle

        while True:
            exist_follower = False
            follower_info=traci.vehicle.getFollower(leader)

            if follower_info[0] == '' or follower_info is None:
                break
            follower, distance = follower_info
            if following_path==self.get_following_path(follower) and traci.vehicle.getTypeID(follower)!="human_vehicle":
                speed = traci.vehicle.getSpeed(follower)
                timeheadway = distance / (speed+0.01) # avoid divided by zero
                if distance <= self.distance_threshold or timeheadway <= self.timeheadway_threshold:
                    vehicles.append(follower)
                    leader=follower
                    exist_follower = True
                    if len(vehicles) >= self.max_platoon_size:
                        break
            if not exist_follower:
                break

        self.vehicles=vehicles


    def update_vehicle_type(self):
        for vehicle in self.vehicles:
            if vehicle==self.leading_vehicle:
                traci.vehicle.setType(vehicle,'leading_vehicle')
            else:
                traci.vehicle.setType(vehicle,'following_vehicle')

    def save_action(self, action):
        self.action = action
        if self.action == 0:
            # 15mph
            self.speed = 6.7
        elif self.action == 1:
            # 20
            self.speed = 8.9
        elif self.action == 2:
            # 25
            self.speed = 11.2
        elif self.action == 3:
            # 30
            self.speed = 13.4
        elif self.action == 4:
            # 35
            self.speed = 15.7


    def do_action(self, current_vehicles):
        for vehicle in self.vehicles:
            if vehicle not in current_vehicles:
                return
            traci.vehicle.setMaxSpeed(vehicle, self.speed)

    def get_obs(self):
        if self.condition==1:
            return self.get_obs_for_work_pa()
        else:
            return [-1,-1,-1,-1,-1]

    def get_obs_for_work_pa(self):
        ahead_signal, distance, lane_phase_index  = self.get_ahead_signal()

        vehicle_num = len(self.vehicles)

        speed=traci.vehicle.getSpeed(self.leading_vehicle)
        if ahead_signal is not None:
            phase=traci.trafficlight.getRedYellowGreenState(ahead_signal)
            current_phase=phase_list.index(phase)
        else:
            current_phase=-1

        obs = [distance, vehicle_num, speed, current_phase,lane_phase_index]

        return obs

    def get_ahead_signal(self):

        if self.condition==0:
            return None,0,0

        lane_id=traci.vehicle.getLaneID(self.leading_vehicle)
        length=self.lane_information[lane_id][0]
        ahead_signal=self.lane_information[lane_id][1]
        phase_index=self.lane_information[lane_id][2]

        position=traci.vehicle.getLanePosition(self.leading_vehicle)
        distance=length-position

        return ahead_signal, distance,phase_index

    def update_reward_info(self,current_vehicles):
        fuel_consumption=0
        speed=0
        if self.leading_vehicle in current_vehicles:
            speed=traci.vehicle.getSpeed(self.leading_vehicle)
            fuel_consumption=traci.vehicle.getFuelConsumption(self.leading_vehicle)
        self.fuel_consumption+=fuel_consumption
        self.reward_speed.append(speed)

    def get_reward(self):
        if self.condition==0:
            ave_speed=np.mean(self.reward_speed)
            veh_m=np.sum(self.reward_speed)
            if veh_m<1:
                reward = -30
            else:
                reward = ave_speed - min(self.fuel_consumption/(veh_m*50),30)
            return reward
        else:
            return 0


class Platoon_System(PlatoonAgent):
    def __init__(self,PA_num,max_platoon_size,distance_threshold,timeheadway_threshold,control_interval):
        self.PA_num=PA_num
        self.distance_threshold=distance_threshold
        self.timeheadway_threshold=timeheadway_threshold
        self.max_platoon_size=max_platoon_size
        self.control_interval=control_interval
        self.max_speed=15.7

        self.platoon_id_list=self.init_id_list(PA_num) # depend on system load
        self.lane_information=get_lane_information()

        self.platoon_set={}
        self.platoon_vehicle_dict={}
        self.current_vehicles = []
        self.HV_vehicles = []

    def reset(self):
        self.platoon_set = {}
        self.platoon_vehicle_dict = {}
        self.current_vehicles = []
        self.HV_vehicles = []

    def init_id_list(self,PA_num):
        agent_name=[]
        for i in range(PA_num):
            agent_name.append('platoon'+str(i))
        return agent_name

    def update(self):
        self.update_vehicles()
        self.delete_PA()
        self.update_PA()
        self.add_new_PA()
        self.set_vehicle_type()

    def set_vehicle_type(self):
        grouped_vehicles = [vehicle for sublist in self.platoon_vehicle_dict.values() for vehicle in sublist]
        non_grouped_vehicles = list(set(self.current_vehicles) - set(grouped_vehicles)-set(self.HV_vehicles))

        for v in non_grouped_vehicles:
            traci.vehicle.setType(v,'idle_vehicle')
            traci.vehicle.setMaxSpeed(v, self.max_speed)

        for platoon_id in self.platoon_set:
            platoon=self.platoon_set[platoon_id]
            if platoon.condition==1:
                platoon.update_vehicle_type()

    def update_vehicles(self):
        self.current_vehicles=traci.vehicle.getIDList()

        self.HV_vehicles=[]
        for v in self.current_vehicles:
            if traci.vehicle.getTypeID(v)=='human_vehicle':
                self.HV_vehicles.append(v)

    def update_PA(self):
        leave_platoon_id = []

        for platoon_id in self.platoon_set:
            platoon = self.platoon_set[platoon_id]
            if platoon.leading_vehicle not in self.current_vehicles:
                leave_platoon_id.append(platoon_id)
                continue
            platoon.update_vehicles()
            vehicles=platoon.vehicles
            self.platoon_vehicle_dict.update({platoon_id:vehicles})

        merged_platoon_ids=[]
        for platoon_id in self.platoon_set:
            if platoon_id in leave_platoon_id:
                continue
            platoon=self.platoon_set[platoon_id]
            vehicles=platoon.vehicles
            for other_id  in self.platoon_set:
                if (other_id==platoon_id) or (other_id in leave_platoon_id):
                    continue
                other_platoon = self.platoon_set[other_id]
                if other_platoon.leading_vehicle in vehicles:
                    merged_platoon_ids.append(other_id)

        for platoon_id in leave_platoon_id+merged_platoon_ids:
            self.platoon_vehicle_dict[platoon_id]=[]
            self.platoon_set[platoon_id].condition=0


    def delete_PA(self):
        to_delete=[]
        for platoon_id in self.platoon_set:
            if self.platoon_set[platoon_id].condition==0:
                to_delete.append(platoon_id)
        for platoon_id in to_delete:
            del self.platoon_set[platoon_id]
            #del self.platoon_vehicle_dict[platoon_id]


    def add_new_PA(self):
        remain_vacant_id_list=list(set(self.platoon_id_list)-set(self.platoon_vehicle_dict.keys()))

        grouped_vehicles = [vehicle for sublist in self.platoon_vehicle_dict.values() for vehicle in sublist]
        non_grouped_vehicles=set(self.current_vehicles)-set(grouped_vehicles)-set(self.HV_vehicles)
        new_grouped_vehicles = {}
        leading_set=set()
        while non_grouped_vehicles-leading_set:
            if len(new_grouped_vehicles) >= len(remain_vacant_id_list):
                print("reaches max load")
                break

            leading_vehicle=list(non_grouped_vehicles-leading_set)[0]
            leading_set.add(leading_vehicle)

            following_path = self.get_following_path(leading_vehicle)
            if following_path[1]==-1:
                continue

            vehicles=[leading_vehicle]

            leader =leading_vehicle
            while True:
                exist_follower = False
                follower_info = traci.vehicle.getFollower(leader)
                if follower_info[0]=='' or follower_info is None:
                    break

                follower,distance=follower_info
                if (follower in non_grouped_vehicles) and following_path == self.get_following_path(follower):
                    speed=traci.vehicle.getSpeed(follower)
                    timeheadway=distance / (speed+0.01) # avoid divided by zero
                    if distance <= self.distance_threshold or timeheadway <= self.timeheadway_threshold:
                        vehicles.append(follower)
                        leader = follower
                        exist_follower=True
                        if len(vehicles) >= self.max_platoon_size:
                            break
                if not exist_follower:
                    break

            if len(vehicles) < self.max_platoon_size:
                follower=leading_vehicle
                while True:
                    exist_follower = False
                    leader_info=traci.vehicle.getLeader(follower)
                    if (leader_info is None) or (leader_info[0]==''):
                        break
                    leader,distance=leader_info
                    if (leader in non_grouped_vehicles) and following_path==self.get_following_path(leader):
                        speed = traci.vehicle.getSpeed(follower)
                        timeheadway = distance / (speed+0.01) # avoid divided by zero
                        if distance <= self.distance_threshold or timeheadway <= self.timeheadway_threshold:
                            vehicles=[leader]+vehicles
                            follower=leader
                            exist_follower = True
                            if len(vehicles) >= self.max_platoon_size:
                                break
                    if not exist_follower:
                        break

            if len(vehicles)>=3:
                new_grouped_vehicles[leading_vehicle]=vehicles
                non_grouped_vehicles-=set(vehicles)

        index=0
        for leading_vehicle,vehicles in new_grouped_vehicles.items():
            platoon_id = remain_vacant_id_list[index]

            platoon = PlatoonAgent(platoon_id, leading_vehicle, vehicles,self.max_platoon_size,self.distance_threshold,
                 self.timeheadway_threshold,self.lane_information,self.control_interval)
            self.platoon_set[platoon_id] = platoon
            self.platoon_vehicle_dict[platoon_id] = vehicles
            index+=1