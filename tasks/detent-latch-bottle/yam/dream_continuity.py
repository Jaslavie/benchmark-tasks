"""Run the official CLI with a process-local Dream episode continuity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


class SessionContinuityError(RuntimeError):
    pass


def install_gate(session_class, record, clock=time.monotonic):
    original_start, original_act = session_class.start, session_class.act

    def start(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        self._continuity_state = dict(session=None, count=0, finished=None)
        record(dict(kind='episode_start'))
        return result

    def act(self, *args, **kwargs):
        state = self._continuity_state
        began = clock()
        chunk = original_act(self, *args, **kwargs)
        finished = clock()
        session = chunk.session_id
        fingerprint = hashlib.sha256(session.encode()).hexdigest() if session else None
        previous = state['session']
        continuous = bool(session) and (previous is None or previous == fingerprint)
        record(dict(kind='chunk', index=state['count'] + 1,
                    session_sha256=fingerprint, continuous=continuous,
                    idle_s=None if state['finished'] is None else began-state['finished'],
                    request_s=finished-began, attempts=chunk.attempts))
        if not continuous:
            raise SessionContinuityError('Dream server session changed or is missing; trial invalid')
        state.update(session=fingerprint, count=state['count']+1, finished=finished)
        return chunk

    session_class.start, session_class.act = start, act


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--continuity-log', required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    from inspect_robots_dream.session import DreamSession
    with args.continuity_log.open('x', buffering=1) as handle:
        install_gate(DreamSession, lambda row: handle.write(json.dumps(row)+'\n'))
        sys.argv = ['inspect-robots', *forwarded]
        from inspect_robots.cli import main as cli_main
        return cli_main()


if __name__ == '__main__':
    main()
