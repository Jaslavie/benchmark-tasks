"""Contact sweep scorer. Physical probe displacement supplies every reading."""
from dataclasses import dataclass
import math


@dataclass
class Sample:
    t: float
    traverse: float
    probe_q: float
    yaw: float
    centre_error: float
    tip_contact: bool
    pad_contact: bool
    clamp_compression: float
    adjustment_hand_clear: bool
    both_open_clear: bool
    speed: float


class TrammingScore:
    def __init__(self):
        self.invalid=False
        self.segment=[]
        self.best_inspection=0.
        self.verified=None
        self.hold=None
        self.last_t=None
        self.value=0.
        self.last_range=None

    def update(self,m):
        nums=(m.t,m.traverse,m.probe_q,m.yaw,m.centre_error,m.clamp_compression,m.speed)
        if not all(math.isfinite(v) for v in nums) or m.centre_error>.0005:
            self.invalid=True
        if self.invalid:
            self.value=0.;self.hold=None;return 0.
        continuous=self.last_t is None or 0 <= m.t-self.last_t <= .100001
        self.last_t=m.t
        clamped=m.pad_contact and .00025 <= m.clamp_compression <= .0055
        valid=m.tip_contact and -.006 <= m.probe_q <= .0005 and abs(m.traverse)<=.040
        if not continuous:
            self.segment=[];self.hold=None
        if self.verified is not None:
            if (not clamped or abs(m.yaw-self.verified['yaw'])>math.radians(.1)
                    or abs(m.centre_error-self.verified['centre_error'])>.0001):
                self.verified=None;self.hold=None
        if valid:
            self.segment.append(m)
            # A stopped dial is not a measurement. Both endpoints and the centre
            # must occur in the same continuous valid contact interval.
            ys=[p.traverse for p in self.segment]
            full=(min(ys)<=-.030 and max(ys)>=.030
                  and any(abs(p.traverse)<=.002 for p in self.segment))
            if full:
                span=max(p.probe_q for p in self.segment)-min(p.probe_q for p in self.segment)
                self.last_range=span
                self.best_inspection=max(self.best_inspection,.5 if span<=.0005 else .25)
                ysaw=[p.yaw for p in self.segment]
                centres=[p.centre_error for p in self.segment]
                released=all(p.adjustment_hand_clear for p in self.segment)
                all_clamped=all(p.pad_contact and .00025<=p.clamp_compression<=.0055 for p in self.segment)
                if (span<=.0002 and released and all_clamped
                        and max(ysaw)-min(ysaw)<=math.radians(.1)
                        and max(centres)-min(centres)<=.0001):
                    self.verified=dict(yaw=m.yaw,centre_error=m.centre_error,range_m=span,t=m.t)
                # Preserve the endpoint sample so the reverse pass is independent.
                self.segment=[m]
        else:
            self.segment=[]
        if self.verified is not None and clamped:
            if m.both_open_clear and m.speed<=.008:
                if self.hold is None:self.hold=m.t
                self.value=1. if m.t-self.hold>=1.-1e-9 else .75
            else:
                self.hold=None;self.value=.75
        else:
            self.hold=None;self.value=self.best_inspection
        return self.value
