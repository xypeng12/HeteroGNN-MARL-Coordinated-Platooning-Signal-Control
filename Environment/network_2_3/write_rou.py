import xml.etree.ElementTree as ET
import random
from xml.dom.minidom import parseString
import sumolib  # 需要安装 sumo-tools


def read_net_edges(net_file):
    """使用 sumolib 读取网络中的所有edge ID."""
    net = sumolib.net.readNet(net_file)
    start_edges = [edge.getID() for edge in net.getEdges() if edge.getID().startswith('np')]

    end_edges = [edge.getID() for edge in net.getEdges() if edge.getID().split('_')[-1].startswith('np')]

    return net, start_edges, end_edges


def compute_route(net, start_edge, end_edge):
    """计算从起始边到结束边的最短路线。"""
    from_edge = net.getEdge(start_edge)
    to_edge = net.getEdge(end_edge)
    if from_edge and to_edge:
        route = net.getShortestPath(from_edge, to_edge)
        if route:
            return route[0]  # 返回边列表
    return []  # 如果没有找到路径或输入边无效，返回空列表

def create_rou_xml(net_file, output_file,CAV_penetration_rates=1,demand_rates=1):
    net, start_edges,end_edges = read_net_edges(net_file)

    # 创建XML文档结构
    routes = ET.Element('routes')

    # 定义车辆类型
    types = [
        ('idle_vehicle', 'carFollowModel="Krauss",tau="1.5"'),
        ('human_vehicle', 'carFollowModel="Krauss", tau="1.5"'),
        ('leading_vehicle', 'carFollowModel="Krauss",tau="1.5"'),
        ('following_vehicle', 'carFollowModel="EIDM",tau="1.0"')]

    for vehicle_type, attributes in types:
        ET.SubElement(routes, 'vType', id=vehicle_type, accel="2.6", decel="4.5", sigma="0.5", length="5.0",
                      minGap="2.5", maxSpeed='17.88',**eval(f"dict({attributes})"))

    # 设置车辆类型比例
    vehicles = []
    init_veh_num=600
    veh_num=int(init_veh_num*demand_rates)
    for i in range(veh_num):
        depart_time = random.uniform(0, 800)
        veh_type = 'idle_vehicle' if random.random() < CAV_penetration_rates else 'human_vehicle'
        start_edge = random.choice(start_edges)
        end_edge = random.choice(end_edges)
        route_edges = compute_route(net, start_edge, end_edge)
        edge_ids = ' '.join(edge.getID() for edge in route_edges)

        vehicle = {'id': f"{veh_type}_{i}", 'type': veh_type, 'depart': depart_time, 'edges': edge_ids}
        vehicles.append(vehicle)

    # 按发车时间排序
    vehicles.sort(key=lambda x: x['depart'])

    # 将车辆信息添加到XML树
    for vehicle in vehicles:
        veh_elem = ET.SubElement(routes, 'vehicle', id=vehicle['id'], type=vehicle['type'],
                                 depart=str(vehicle['depart']))
        ET.SubElement(veh_elem, 'route', edges=vehicle['edges'])

    # 美化和写入文件
    rough_string = ET.tostring(routes, 'utf-8')
    reparsed = parseString(rough_string)
    with open(output_file, 'w') as f:
        f.write(reparsed.toprettyxml(indent="    "))


create_rou_xml('network23.net.xml', 'network23.rou.xml', CAV_penetration_rates=1,demand_rates=1)
# 使用示例
#for different_partition_rates:
rate_list=[0,0.2,0.4,0.6,0.8,1.0]
for CAV_penetration_rates in rate_list:
    create_rou_xml('network23.net.xml', 'network23_CAV-rate%.1f.rou.xml' % CAV_penetration_rates, CAV_penetration_rates=CAV_penetration_rates,demand_rates=1)

demand_list=[0.2,0.4,0.6,0.8,1.0,1.2,1.4]
for demand_rates in demand_list:
    create_rou_xml('network23.net.xml', 'network23_demand-rate%.1f.rou.xml' % demand_rates, CAV_penetration_rates=1,demand_rates=demand_rates)