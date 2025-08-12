import xml.etree.ElementTree as et
filename='Environment/network_2_3/network23.net.xml'

phase_list=['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr']

NEAREST_PLATOON_NUMBER=3

def get_neighbor_signal(name):
    junctions=read_signal_id(top=False)
    tree = et.parse(filename)
    root = tree.getroot()
    junction_neighbors = []

    for edge in root.findall('edge'):
        if (edge.get('from')==name and edge.get('to') in junctions):
            junction_neighbors.append(edge.get('to'))
        if (edge.get('to') == name and edge.get('from') in junctions):
            junction_neighbors.append(edge.get('from'))

    return junction_neighbors


def read_signal_id(filename=filename,top=True):
    tree = et.parse(filename)
    root= tree.getroot()
    signal_id_list = []
    for signal in root.findall('junction'):
        if signal.get('type') == "traffic_light":
            if top==True:
                signal_id = "signal"+signal.get('id')
            else:
                signal_id = signal.get('id')

            signal_id_list.append(signal_id)
    return signal_id_list

def get_phase_set(name):

    phase_set=phase_list
    return phase_set

def get_controlled_lanes(name):
    tree = et.parse(filename)
    root = tree.getroot()
    controlled_lanes = []

    # 解析相邻junction的信息
    for edge in root.findall('edge'):
        if edge.get('to') == name:
            for lane in edge.findall("lane"):
                controlled_lanes.append(lane.get('id'))

    return controlled_lanes


def get_state_space():
    junctions=read_signal_id(top=False)
    obs_space={}
    for name in junctions:
        lane_number=len(get_controlled_lanes(name))
        obs_space["signal"+name]=2*lane_number
    return obs_space

def sa_get_relevant_agent():
    junctions = read_signal_id(top=False)
    relevant_agent={}
    for name in junctions:
        relevant_agent["signal" + name] =[ "signal" +item for item in get_neighbor_signal(name)]

    return relevant_agent,NEAREST_PLATOON_NUMBER

def get_lane_information():
    tree = et.parse(filename)
    root = tree.getroot()

    lane_information = {}
    junctions = {j.get('id'): j.get('type') for j in root.findall('junction')}


    for edge in root.findall('edge'):
        to_junction = edge.get('to')
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            length = float(lane.get('length'))
            # 检查终点是否为信号节点
            if to_junction in junctions and junctions[to_junction] == 'traffic_light':
                signal_node = to_junction
            else:
                signal_node = None
            lane_information[lane_id]=[length,signal_node]

    north_list = ['np1_nt1_0', 'np2_nt2_0', 'np3_nt3_0', 'nt1_nt4_0', 'nt2_nt5_0', 'nt3_nt6_0']
    south_list = ['nt4_nt1_0', 'nt5_nt2_0', 'nt6_nt3_0', 'np8_nt4_0', 'np9_nt5_0', 'np10_nt6_0']

    west_left_list = ['np4_nt1_1', 'nt1_nt2_1', 'nt2_nt3_1', 'np6_nt4_1', 'nt4_nt5_1', 'nt5_nt6_1']
    west_straightright_list = ['np4_nt1_0', 'nt1_nt2_0', 'nt2_nt3_0', 'np6_nt4_0', 'nt4_nt5_0', 'nt5_nt6_0']
    east_left_list = ['nt2_nt1_1', 'nt3_nt2_1', 'np5_nt3_1', 'nt5_nt4_1', 'nt6_nt5_1', 'np7_nt6_1']
    east_straightright_list = ['nt2_nt1_0', 'nt3_nt2_0', 'np5_nt3_0', 'nt5_nt4_0', 'nt6_nt5_0', 'np7_nt6_0']

    for lane_id in lane_information:
        if lane_id in north_list or lane_id in south_list:
            lane_information[lane_id].append(0)

        elif lane_id in west_left_list or lane_id in east_left_list:
            lane_information[lane_id].append(2)

        elif lane_id in west_straightright_list or lane_id in east_straightright_list:
            lane_information[lane_id].append(4)

        else:
            lane_information[lane_id].append(-1)

    return lane_information

if __name__ == "__main__":
    print(get_phase_set('nt2'))
    print(get_phase_set('nt3'))
    print(get_phase_set('nt4'))
    print(get_phase_set('nt5'))
    print(get_phase_set('nt6'))
