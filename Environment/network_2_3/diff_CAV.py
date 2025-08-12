import xml.etree.ElementTree as ET
import random

# Load the .rou file
tree = ET.parse('network23.rou.xml')
root = tree.getroot()

# CAV penetration rates
CAV_penetration_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for rate in CAV_penetration_rates:
    # Create a copy of the root for each rate
    new_root = ET.Element(root.tag, root.attrib)
    for child in root:
        if child.tag == 'vType':
            new_root.append(child)
        elif child.tag == 'vehicle':
            vehicle_type = 'idle_vehicle' if random.random() < rate else 'human_vehicle'
            vehicle = ET.Element(child.tag, attrib=child.attrib)
            vehicle.set('type', vehicle_type)
            for sub_child in child:
                vehicle.append(sub_child)
            new_root.append(vehicle)

    # Write the new .rou file
    new_tree = ET.ElementTree(new_root)
    new_tree.write('network23_CAV-rate%.1f.rou.xml' % rate, encoding='utf-8', xml_declaration=True)