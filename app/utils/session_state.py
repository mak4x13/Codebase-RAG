
import uuid
class SessionState:
    def __init__(self):
        self.repo_id = None
        self.repo_url = None
        self.session_id = uuid.uuid4().hex
