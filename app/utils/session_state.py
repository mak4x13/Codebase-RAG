
import uuid
class SessionState:
    def __init__(self):
        self.repo_id = None
        self.repo_url = None
        self.profile_metadata_id = None
        self.repo_ids = []
        self.repo_name_map = {}
        self.repo_summaries = {}
        self.repo_id_to_name = {}
        self.session_id = uuid.uuid4().hex
