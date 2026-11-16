import unittest
from types import SimpleNamespace
from dream_continuity import install_gate, SessionContinuityError


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        class Session:
            def start(self): pass
            def act(self, response):
                if isinstance(response, Exception): raise response
                return response
        self.records = []
        install_gate(Session, self.records.append)
        self.session = Session()
        self.session.start()

    def chunk(self, session):
        return SimpleNamespace(session_id=session, attempts=1, rows=object())

    def test_same_session_preserves_actions(self):
        response = self.chunk('a')
        self.assertIs(self.session.act(response), response)
        self.assertIs(self.session.act(response), response)
        self.assertTrue(all(r['continuous'] for r in self.records if r['kind']=='chunk'))

    def test_changed_session_fails_before_returning_actions(self):
        self.session.act(self.chunk('a'))
        with self.assertRaises(SessionContinuityError): self.session.act(self.chunk('b'))
        self.assertFalse(self.records[-1]['continuous'])

    def test_missing_session_fails(self):
        with self.assertRaises(SessionContinuityError): self.session.act(self.chunk(None))

    def test_reset_allows_new_episode(self):
        self.session.act(self.chunk('a'))
        self.session.start()
        self.session.act(self.chunk('b'))

    def test_original_error_propagates(self):
        error = ValueError('service unavailable')
        with self.assertRaises(ValueError) as caught: self.session.act(error)
        self.assertIs(caught.exception, error)


if __name__ == '__main__': unittest.main()
