"""CPU geometry and adversarial scoring plus optional official-source IK."""
from dataclasses import replace
import json,math,os,sys
from pathlib import Path
import numpy as np
import design as d
from scoring import Sample,TrammingScore
from native_contacts import any_close


def sample(t,y=0.,**kwargs):
    return replace(Sample(t,y,-.003,0.,0.,True,True,.003,True,True,0.),**kwargs)


def sweep(s,start=0.,yaw=0.,**kwargs):
    result=[]
    for i,y in enumerate(np.linspace(-.035,.035,41)):
        result.append(s.update(sample(start+i*.05,float(y),probe_q=d.predicted_probe_q(float(y),yaw),yaw=yaw,**kwargs)))
    return result


def main():
    checks=[]
    # The cam loads a real compressible shoe. At the open stop it clears the rotor.
    assert d.cam_bottom(d.CAM_OPEN)>d.cam_bottom(d.CAM_CLOSED)+.003
    assert .002<d.expected_clamp_compression(d.CAM_CLOSED)<.0055
    open_gap=d.follower_position(d.CAM_OPEN)[2]-.00575-.001-.0162
    assert open_gap>.0003
    assert d.cam_bottom(d.CAM_CLOSED)>d.cam_bottom(math.radians(-45))
    checks.append('Open shoe clearance and positive overcentre spring compression')
    shoe_half = np.asarray(d.PAD_PARTS[0]['size']) / 2
    support_half_width = d.ROTOR_PARTS[1]['size'][1] / 2
    for angle in np.linspace(-8., 8., 65):
        to_rotor = d.rz(-math.radians(angle))
        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                corner = np.asarray(d.PAD_HOME) + shoe_half * [x_sign, y_sign, 0]
                assert (to_rotor @ corner)[1] < -support_half_width - .001
    checks.append('Clamping shoe clears the raised support throughout swivel travel')
    q=[d.predicted_probe_q(y,d.INITIAL_YAW) for y in (-.035,0,.035)]
    assert min(q)>d.Q_MIN+.0005 and max(q)<d.Q_MAX-.0005
    assert max(q)-min(q)>.004
    assert abs(d.predicted_probe_q(-.035,0)-d.predicted_probe_q(.035,0))<1e-12
    checks.append('Physical jaw plane produces millimetre error at the crooked start and zero ideal aligned sweep')
    s=TrammingScore();assert max(s.update(sample(i*.05)) for i in range(50))==0
    s=TrammingScore();assert max(sweep(s,yaw=d.INITIAL_YAW))==.25
    s=TrammingScore();r=sweep(s);assert max(r)<=.75
    for i in range(41,65):s.update(sample(i*.05,y=.035))
    assert s.value==1.
    checks.append('No centre only success and measured misalignment partial credit and one second full hold')
    for kwargs in ({'tip_contact':False},{'probe_q':d.Q_MIN},{'probe_q':d.Q_MAX}):
        s=TrammingScore()
        for i,y in enumerate(np.linspace(-.035,.035,41)):s.update(sample(i*.05,float(y),**kwargs))
        assert s.value==0
    for kwargs in ({'pad_contact':False},{'clamp_compression':0.},{'adjustment_hand_clear':False}):
        s=TrammingScore();sweep(s,**kwargs)
        for i in range(41,65):s.update(sample(i*.05,y=.035,**kwargs))
        assert s.value<=.5
    checks.append('Missing contact and plunger stop and unlocked or hand held aligned vise cannot pass')
    s=TrammingScore();sweep(s)
    s.update(sample(2.05,y=.035,yaw=math.radians(.2)))
    assert s.verified is None
    for kwargs in ({'centre_error':.001},{'probe_q':float('nan')}):
        s=TrammingScore();assert s.update(sample(0,**kwargs))==0
        assert max(sweep(s,start=.05))==0
    checks.append('Base drift invalidates the final measurement and nonfinite data fails permanently')
    assert any_close([.00001],[1],[0],.0001)
    assert not any_close([.001],[1],[0],.0001)
    checks.append('Native contact uses separation and does not require force sensors')
    root=Path(__file__).resolve().parent
    report=dict(status='CPU only',passed=checks,native_demo=None,dream_run=None)
    yam=Path(os.environ.get('YAM_SOURCE','/mnt/d/robocurve - evals/yam-isaac-lab-arena'))
    if yam.is_dir():
        sys.path[:0]=[str(root),str(root/'runtime'),str(yam)]
        from yam_arena.task_file import load
        from yam_arena.demo import ArmPlan
        from yam_arena import station
        task=load(root/'task.py')
        inventory=[]
        for item in task.module.OBJECTS:
            item._check()
            inventory.append(dict(name=item.name,root_position=item.pos,mass=item.mass,parts=item.parts,
                links=item.links,joints=item.joints,fixed_base=item.fixed_base,self_collisions=item.self_collisions))
        (root/'scene_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        report['ik']={}
        for arm,points in task.demo().items():
            plan=ArmPlan(points,station.BASES[arm])
            for t in np.arange(.05,plan.duration+.05,.05):plan.action7(float(t))
            report['ik'][arm]=dict(worst_mm=plan.worst_ik_mm,worst_degrees=plan.worst_ik_deg,duration_s=plan.duration)
            assert plan.worst_ik_mm<2 and plan.worst_ik_deg<1
            assert plan.duration+3<120
        checks.append('Official source task load and both complete 20 Hz IK trajectories')
    (root/'cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
