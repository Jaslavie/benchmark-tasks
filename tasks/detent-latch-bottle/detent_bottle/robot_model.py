"""Panda arms and worktable, using robosuite 1.4.1 robot assets."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

BAR_Z = .964

def scene(seed, scene_kind="balance"):
    if scene_kind not in ("balance", "table_random", "table_visual"):
        raise ValueError("Unknown dual-arm scene")
    rng = np.random.default_rng(seed)
    offset = float(rng.choice([-1, 1]) * rng.uniform(.035, .085))
    root = ET.fromstring('''<mujoco model="dual_panda_balance">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" integrator="implicitfast" cone="elliptic" iterations="100"/>
      <size nconmax="500" njmax="3000"/>
      <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
      <default><geom friction="1.1 .01 .001" solref=".008 1" solimp=".95 .99 .001"/><joint armature=".02"/></default>
      <asset><texture name="floor_tex" type="2d" builtin="checker" rgb1=".12 .15 .19" rgb2=".16 .19 .23" width="256" height="256"/><material name="floor_mat" texture="floor_tex" texrepeat="5 5"/></asset>
      <worldbody>
        <light pos="0 -1 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
        <light pos="1 1 2" dir="-1 -1 -1" diffuse=".5 .5 .5"/>
        <geom name="floor" type="plane" size="3 3 .1" material="floor_mat"/>
        <geom name="table" type="box" pos="0 0 .76" size=".72 .82 .04" rgba=".28 .32 .38 1"/>
        <body name="post" pos=".20 0 .9075"><geom name="post_geom" type="box" size=".07 .012 .1075" rgba=".08 .12 .17 1"/></body>
        <body name="cradle_left" pos="0 -.25 .875"><geom name="cradle_left_geom" type="box" size=".07 .025 .075" rgba=".4 .45 .5 1"/></body>
        <body name="cradle_right" pos="0 .25 .875"><geom name="cradle_right_geom" type="box" size=".07 .025 .075" rgba=".4 .45 .5 1"/></body>
      </worldbody><actuator/><sensor/><contact/>
    </mujoco>''')
    bar_pos=np.array([0.,0.,BAR_Z]); yaw=0.
    if scene_kind in ('table_random','table_visual'):
        # Separate random stream: visible placement does not determine the hidden CoM.
        pose_rng=np.random.default_rng(np.random.SeedSequence([seed, 8127]))
        bar_pos=np.array([pose_rng.uniform(-.08,.01),pose_rng.uniform(-.045,.045),.814])
        yaw=pose_rng.uniform(-np.pi/7.2,np.pi/7.2)  # ±25 degrees, reachable workspace.
        world=root.find('worldbody')
        for side in ('left','right'): world.remove(world.find(f"body[@name='cradle_{side}']"))
    spec = importlib.util.find_spec('robosuite')
    assets = Path(spec.origin).parent/'models/assets/robots/panda'
    for side, y, color in [('left', -.42, '.1 .55 .95 1'), ('right', .42, '.95 .5 .1 1')]:
        robot = ET.parse(assets/'no_texture_robot.xml').getroot()
        for node in robot.iter():
            for attr in ('name', 'joint', 'mesh', 'material', 'site', 'body'):
                if attr in node.attrib:
                    node.set(attr, side+'_'+node.get(attr))
            if node.tag == 'mesh': node.set('file', str(assets/node.get('file')))
        for geom in robot.iter('geom'):
            if geom.get('name', '').endswith('_collision'): geom.set('rgba', '0 0 0 0')
        base = robot.find('worldbody/body')
        base.set('pos', f'-.40 {y} .8')
        hand = base.find(f'.//body[@name="{side}_right_hand"]')
        ET.SubElement(hand, 'site', name=side+'_wrist', size='.008', rgba=color)
        ET.SubElement(hand, 'geom', name=side+'_palm', type='box', pos='0 0 .045', size='.055 .025 .02', rgba=color)
        ET.SubElement(hand, 'site', name=side+'_tip', pos='0 0 .12', size='.004', rgba=color)
        for finger, sign in [('a', -1), ('b', 1)]:
            name=f'{side}_finger_{finger}'
            body=ET.SubElement(hand, 'body', name=name)
            ET.SubElement(body, 'joint', name=name, type='slide', axis=f'{sign} 0 0', range='0 .045', damping='4', armature='.001')
            ET.SubElement(body, 'geom', name=name+'_pad', type='box', pos=f'{sign*.008} 0 .11', size='.008 .016 .035', mass='.04', rgba='.18 .2 .23 1', friction='1.5 .01 .001')
            ET.SubElement(body, 'site', name=name+'_touch', type='box', pos=f'{sign*.001} 0 .11', size='.003 .017 .036', rgba='0 0 0 0')
            ET.SubElement(root.find('sensor'), 'touch', name=name+'_normal', site=name+'_touch')
        # Wrist inertia retained; fingers add real moving mass. No grasp weld.
        for typ in ('force', 'torque'):
            ET.SubElement(root.find('sensor'), typ, name=side+'_'+typ, site=side+'_wrist')
        root.find('worldbody').append(base)
        root.find('asset').extend(robot.find('asset'))
        root.find('actuator').extend(robot.find('actuator'))
    # Append after all fourteen arm motors so their existing indices stay stable.
    for side in ('left', 'right'):
        for finger in ('a', 'b'):
            name=f'{side}_finger_{finger}'
            ET.SubElement(root.find('actuator'), 'position', name=name, joint=name, kp='400', ctrlrange='0 .045', forcerange='-12 12')
    # A 0.1 kg shell plus concealed 0.5 kg ballast, merged analytically.
    mass, shell, ballast = .6, .1, .5
    b = offset*mass/ballast
    inertia_shell = shell/3*np.array([.30**2+.014**2, .035**2+.014**2, .035**2+.30**2])
    inertia_ballast = ballast/3*np.array([.015**2+.01**2, .025**2+.01**2, .025**2+.015**2])
    I = inertia_shell + inertia_ballast + np.array([1,0,1])*(shell*offset**2+ballast*(b-offset)**2)
    bar = ET.SubElement(root.find('worldbody'), 'body', name='bar', pos=' '.join(map(str,bar_pos)), quat=f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}')
    ET.SubElement(bar, 'freejoint', name='bar_joint')
    ET.SubElement(bar, 'inertial', mass=str(mass), pos=f'0 {offset} 0', diaginertia=' '.join(map(str,I)))
    ET.SubElement(bar, 'geom', name='bar_geom', type='box', size='.035 .30 .014', rgba='.72 .76 .8 1')
    ET.SubElement(bar, 'geom', name='center_mark', type='box', pos='0 0 .0145', size='.035 .002 .0005', rgba='.15 .2 .25 1', contype='0', conaffinity='0', mass='0')
    from scipy.spatial.transform import Rotation
    for name, p, look in [('agentview',[1.3,-1.55,2.0],[-.1,0,1.05]), ('wristview',[.65,0,2.4],[0,0,.95])]:
        z=np.array(p)-look; z/=np.linalg.norm(z)
        x=np.cross([0,0,1],z); x/=np.linalg.norm(x); yy=np.cross(z,x)
        q=Rotation.from_matrix(np.column_stack([x,yy,z])).as_quat()[[3,0,1,2]]
        ET.SubElement(root.find('worldbody'),'camera',name=name,pos=' '.join(map(str,p)),quat=' '.join(map(str,q)),fovy='48')
    return ET.tostring(root,encoding='unicode'), offset
