import xmltodict
import os


def find_joint(d, name):
    for j in d['robot']['joint']:
        if j['@name'] == name:
            return j
    return None


def find_link(d, name):
    for l in d['robot']['link']:
        if l['@name'] == name:
            return l
    return None


# generate Trans profile
# template (fixed) :   
# <NAME> (fixed) <origin rpy="{R} {P} {Y}" xyz="{X} {Y} {Z}"/>
# tempT.setProfile(x,y,z,R,P,Y);

# template (active): (usually, gcIndex = gvIndex+1)
# <<NAME>> ({rev/prsm}) <axis xyz="{X} {Y} {Z}"/> (gc[13])
# tempT.setProfile(x,y,z,R,P,Y,newTyp,ax,ay,az,gcIndex,gvIndex);

def generate_trans_profile(d, name, gcIdx=-1, pSpace=2, target_name="tempT"):
    j_raw = find_joint(d, name)

    typ = j_raw['@type'][0]
    name = j_raw['@name']

    [x, y, z] = j_raw['origin']['@xyz'].split(" ")
    [R, P, Y] = j_raw['origin']['@rpy'].split(" ")
    s_origin = f'<origin rpy="{R} {P} {Y}" xyz="{x} {y} {z}"/>'

    if typ != 'f':
        [ax, ay, az] = j_raw['axis']['@xyz'].split(" ")
        s_axis = f'<axis xyz="{ax} {ay} {az}/>" (gc[{gcIdx}])'

    s_comment = ""
    s_profile = ""

    if typ == 'f':
        s_comment = f'{" " * pSpace}// <{name}> ({j_raw["@type"]}) {s_origin}'
        s_profile = f'{" " * pSpace}{target_name}.setProfile({x}, {y}, {z},   {R}, {P}, {Y});'

    else:
        s_comment = f'{" " * pSpace}// <<{name}>> ({j_raw["@type"]}) {s_origin} {s_axis}'
        s_profile = f'{" " * pSpace}{target_name}.setProfile({x}, {y}, {z},   {R}, {P}, {Y}, \'{typ}\',  {ax}, {ay}, {az}, {gcIdx}, {gcIdx - 1});'

    return f'{s_comment}\n{s_profile}'


# generate Inertia profile
# template:
# // [LH_HAA]
# // <origin rpy="0 0 0" xyz="-0.063 7e-05 0.00046"/>
# // <mass value="2.04"/>
# // <inertia ixx="0.001053013" ixy="4.527e-05" ixz="8.855e-05" iyy="0.001805509" iyz="9.909e-05" izz="0.001765827"/>
# tempI.setProfile(x,y,z,R,P,Y,mass,ixx, ixy, ixz, iyy, iyz, izz)

def generate_inertia_profile(d, name, pSpace=2):
    l_raw = find_link(d, name)
    l_inert = l_raw['inertial']
    l_r = l_inert['inertia']

    name = l_raw['@name']
    [x, y, z] = l_inert['origin']['@xyz'].split(" ")
    [R, P, Y] = l_inert['origin']['@rpy'].split(" ")
    mass = l_inert['mass']['@value']
    [ixx, ixy, ixz, iyy, iyz, izz] = [l_r['@ixx'], l_r['@ixy'], l_r['@ixz'], l_r['@iyy'], l_r['@iyz'], l_r['@izz']]

    s_label = f'{" " * pSpace}// [{name}] <origin rpy="{R} {P} {Y}" xyz="{x} {y} {z}"/>'
    s_inertia = f'{" " * pSpace}// mass value="{mass}"/> <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyy}" izz="{izz}"/>'
    s_profile = f'{" " * pSpace}tempI.setProfile({x},{y},{z},  {R},{P},{Y},  {mass},  {ixx}, {ixy}, {ixz}, {iyy}, {iyz}, {izz});'

    return f'{s_label}\n{s_inertia}\n{s_profile}'


def generate_leg_trans(robot_dict, leg):
    names = [f'base_{leg}_HAA', f'{leg}_HAA', f'{leg}_HIP_{leg}_hip_fixed', f'{leg}_hip_fixed_{leg}_HFE', f'{leg}_HFE',
             f'{leg}_THIGH_{leg}_thigh_fixed', f'{leg}_thigh_fixed_{leg}_KFE', f'{leg}_KFE',
             f'{leg}_shank_{leg}_shank_fixed', f'{leg}_shank_fixed_{leg}_FOOT']
    joint_gc = {
        "LF": [-1, 7, -1, -1, 8, -1, -1, 9, -1, -1],
        "RF": [-1, 10, -1, -1, 11, -1, -1, 12, -1, -1],
        "LH": [-1, 13, -1, -1, 14, -1, -1, 15, -1, -1],
        "RH": [-1, 16, -1, -1, 17, -1, -1, 18, -1, -1]
    }
    gc = joint_gc[leg]

    entries = []
    for i in range(len(names)):
        entries.append(generate_trans_profile(robot_dict, names[i], gc[i], 0))
    print(
        f"""
auto {leg}Hip = new Link("{leg}_HIP",'a');
{entries[0]}
{leg}Hip -> addTrans(tempT);

{entries[1]}
{leg}Hip -> addTrans(tempT);
robot.addLink({leg}Hip,base);

auto {leg}Thigh = new Link("{leg}_THIGH",'a');
{entries[2]}
{leg}Thigh -> addTrans(tempT);

{entries[3]}
{leg}Thigh -> addTrans(tempT);

{entries[4]}
{leg}Thigh -> addTrans(tempT);
robot.addLink({leg}Thigh,{leg}Hip);

auto {leg}Shank = new Link("{leg}_SHANK",'a');
{entries[5]}
{leg}Shank -> addTrans(tempT);

{entries[6]}
{leg}Shank -> addTrans(tempT);

{entries[7]}
{leg}Shank -> addTrans(tempT);
robot.addLink({leg}Shank,{leg}Thigh);

auto {leg}Foot = new Link("{leg}_FOOT",'e');
{entries[8]}
{leg}Foot -> addTrans(tempT);

{entries[9]}
{leg}Foot -> addTrans(tempT);
robot.addLink({leg}Foot,{leg}Shank);
        """
    )

def generate_leg_inertia(robot_dict,leg):
    elements = [
        ('t',f'base_{leg}_HAA'),
        ('i',f'{leg}_HAA'),

        ('i',f'{leg}_HIP'),
        ('t',f'{leg}_HIP_{leg}_hip_fixed'),
        ('i',f'{leg}_hip_fixed'),
        ('tt',f'{leg}_hip_fixed_{leg}_HFE'),
        ('i',f'{leg}_HFE'),

        ('i',f'{leg}_THIGH'),
        ('t',f'{leg}_THIGH_{leg}_thigh_fixed'),
        ('i',f'{leg}_thigh_fixed'),
        ('tt',f'{leg}_thigh_fixed_{leg}_KFE'),
        ('i',f'{leg}_KFE'),

        ('i',f'{leg}_SHANK'),
        ('t',f'{leg}_shank_{leg}_shank_fixed'),
        ('i',f'{leg}_shank_fixed'),
        ('tt',f'{leg}_shank_fixed_{leg}_FOOT'),
        ('i',f'{leg}_FOOT'),
    ]

    entries = []
    for (typ,name) in elements:
        s = ""
        if typ == 'i':
            s = generate_inertia_profile(robot_dict,name,0)
        elif typ == 't':
            s = generate_trans_profile(robot_dict,name,-1,0)
        elif typ == 'tt':
            s = generate_trans_profile(robot_dict,name,-1,0,"tempTT")
            s += "\ntempT.attachTrans(tempTT);"
        entries.append(s)

    # for s in entries:
    #     print(s)
    print(f"""
//---attached to BASE ---
// (base --"base_{leg}_HAA" --> {leg}_HAA) (base added later)
{entries[0]}
{entries[1]}
robot.getLinkByName("BASE")->addInertia(tempI.expressedIn(tempT));

//---attached to {leg}_HIP ---
// ({leg}_HIP --"{leg}_HIP_{leg}_hip_fixed"--> {leg}_hip_fixed --> "{leg}_hip_fixed_{leg}_HFE" --> {leg}_HFE)
{entries[2]}
robot.getLinkByName("{leg}_HIP")->addInertia(tempI);

{entries[3]}
{entries[4]}
robot.getLinkByName("{leg}_HIP")->addInertia(tempI.expressedIn(tempT));

{entries[5]}
{entries[6]}
robot.getLinkByName("{leg}_HIP")->addInertia(tempI.expressedIn(tempT));

//---attached to {leg}_THIGH ---
// ({leg}_THIGH --"{leg}_THIGH_{leg}_thigh_fixed"--> {leg}_thigh_fixed --> "{leg}_thigh_fixed_{leg}_KFE" --> {leg}_KFE)
{entries[7]}
robot.getLinkByName("{leg}_THIGH")->addInertia(tempI);

{entries[8]}
{entries[9]}
robot.getLinkByName("{leg}_THIGH")->addInertia(tempI.expressedIn(tempT));

{entries[10]}
{entries[11]}
robot.getLinkByName("{leg}_THIGH")->addInertia(tempI.expressedIn(tempT));

//---attached to {leg}_SHANK ---
// ({leg}_SHANK --"{leg}_SHANK_{leg}_shank_fixed"--> {leg}_shank_fixed --> "{leg}_shank_fixed_{leg}_FOOT" --> {leg}_FOOT)

{entries[12]}
robot.getLinkByName("{leg}_SHANK")->addInertia(tempI);

{entries[13]}
{entries[14]}
robot.getLinkByName("{leg}_SHANK")->addInertia(tempI.expressedIn(tempT));

{entries[15]}
{entries[16]}
robot.getLinkByName("{leg}_SHANK")->addInertia(tempI.expressedIn(tempT));

    """)

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    anymal_path = dir_path + "/../../resource/anymal_c/urdf/anymal.urdf"
    xml_string = open(anymal_path, "r").read()
    # print(xml_string)

    robot_dict = xmltodict.parse(xml_string)
    # print(find_link(robot_dict,"LH_HAA"))
    # print(find_joint(robot_dict,"LH_HAA"))

    # # print(robot_dict)
    # links = robot_dict['robot']['link']
    # joints = robot_dict['robot']['joint']

    # print("\n\n----LINKS----\n")

    # for link in links:
    #     print(link['@name'])
    #     try:
    #         print(link['inertial'])
    #     except:
    #         print("(no inertial property)")

    # print("\n\n----JOINTS----\n")

    # for j in robot_dict['robot']['joint']:
    #     print(j['@name'])
    #     print(j.keys())
    #     print(j['@type'] + " ")
    #     print(j['origin']['@xyz'].split(" "))
    #     try:
    #         print(j['axis']['@xyz'].split(" "))
    #     except:
    #         print("no actuated axis")

    # print(generate_trans_profile(robot_dict,"base_to_lidar_cage",13,4))
    print(generate_trans_profile(robot_dict,"base_hatch",13,4))
    print(generate_inertia_profile(robot_dict, "hatch", 4))

    # generate_leg_trans(robot_dict,"RH")
    # generate_leg_inertia(robot_dict, "RH")


main()
