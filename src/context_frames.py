"""
Context-Frames Parser and Relational Reasoning Graph.

Parses Context-Frames (F1 to F20).
Maps frames and relational hierarchies into reasoning nodes, propositions, and evidence.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple
from src.beta_model import BetaDistribution
from src.pln_inference import PLNBelief

CONTEXT_FRAMES_RAW_TEXT = """
(Frame (frameID F1) (parentID NONE) (status active) (priority high) (deliverable "What is the time?") (results "The current time is 10:42 AM.")) 
(Frame (frameID F2) (parentID F1) (status completed) (priority medium) (deliverable "What is the time in Asia?") (results "Asia spans multiple time zones.")) 
(Frame (frameID F3) (parentID NONE) (status active) (priority high) (deliverable "Teach me Python.") (results "Started with variables, data types, and basic syntax.")) 
(Frame (frameID F4) (parentID F3) (status pending) (priority medium) (deliverable "Teach me Python loops.") (results "Not started.")) 
(Frame (frameID F5) (parentID F3) (status completed) (priority low) (deliverable "Give me an example of a Python for loop.") (results "for i in range(5): print(i)")) 
(Frame (frameID F6) (parentID NONE) (status completed) (priority medium) (deliverable "Explain photosynthesis.") (results "Plants convert light energy into chemical energy.")) 
(Frame (frameID F7) (parentID F6) (status active) (priority high) (deliverable "Explain photosynthesis to a 10-year-old.") (results "Plants use sunlight to make their own food.")) 
(Frame (frameID F8) (parentID NONE) (status pending) (priority low) (deliverable "Plan a trip to Japan.") (results "No itinerary generated yet.")) 
(Frame (frameID F9) (parentID F8) (status active) (priority high) (deliverable "Find places to visit in Tokyo.") (results "Tokyo Tower, Shibuya, Asakusa, and Meiji Shrine.")) 
(Frame (frameID F10) (parentID F8) (status pending) (priority medium) (deliverable "Create a three-day Tokyo itinerary.") (results "Waiting for destination preferences.")) 
(Frame (frameID F11) (parentID NONE) (status completed) (priority high) (deliverable "Calculate 25 multiplied by 16.") (results "400")) 
(Frame (frameID F12) (parentID NONE) (status active) (priority medium) (deliverable "Help me learn machine learning.") (results "Outlined supervised, unsupervised, and reinforcement learning.")) 
(Frame (frameID F13) (parentID F12) (status completed) (priority high) (deliverable "Explain supervised learning.") (results "Supervised learning trains models using labeled examples.")) 
(Frame (frameID F14) (parentID F12) (status active) (priority medium) (deliverable "Explain neural networks.") (results "Introduced neurons, layers, weights, and activation functions.")) 
(Frame (frameID F15) (parentID F14) (status pending) (priority low) (deliverable "Give me a simple neural network example.") (results "No example generated yet.")) 
(Frame (frameID F16) (parentID NONE) (status completed) (priority low) (deliverable "Write a short poem about rain.") (results "Rain taps softly against the sleeping street.")) 
(Frame (frameID F17) (parentID NONE) (status active) (priority high) (deliverable "Help me build a website.") (results "Defined an initial HTML, CSS, and JavaScript structure.")) 
(Frame (frameID F18) (parentID F17) (status completed) (priority medium) (deliverable "Create the website navigation bar.") (results "Created a navigation bar with Home, About, and Contact links.")) 
(Frame (frameID F19) (parentID F17) (status active) (priority high) (deliverable "Add a login page to the website.") (results "Created username and password input fields.")) 
(Frame (frameID F20) (parentID F19) (status pending) (priority high) (deliverable "Add password validation to the login page.") (results "Password validation has not been implemented yet."))
"""


@dataclass
class FrameRecord:
    frame_id: str
    parent_id: str
    status: str
    priority: str
    deliverable: str
    results: str

    @property
    def full_text(self) -> str:
        """Normalized textual representation for vector embedding."""
        return f"Frame {self.frame_id}: Deliverable: {self.deliverable}. Results: {self.results}."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frameID": self.frame_id,
            "parentID": self.parent_id,
            "status": self.status,
            "priority": self.priority,
            "deliverable": self.deliverable,
            "results": self.results,
        }


def parse_context_frames(raw_text: str = CONTEXT_FRAMES_RAW_TEXT) -> List[FrameRecord]:
    """Parse raw S-expressions into structured FrameRecord objects."""
    pattern = re.compile(
        r'\(Frame\s+'
        r'\(frameID\s+([A-Za-z0-9_]+)\)\s+'
        r'\(parentID\s+([A-Za-z0-9_]+)\)\s+'
        r'\(status\s+([A-Za-z0-9_]+)\)\s+'
        r'\(priority\s+([A-Za-z0-9_]+)\)\s+'
        r'\(deliverable\s+"([^"]*)"\)\s+'
        r'\(results\s+"([^"]*)"\)\)',
        re.MULTILINE,
    )
    records = []
    for match in pattern.finditer(raw_text):
        f_id, p_id, status, priority, deliverable, results = match.groups()
        records.append(
            FrameRecord(
                frame_id=f_id,
                parent_id=p_id,
                status=status.lower(),
                priority=priority.lower(),
                deliverable=deliverable,
                results=results,
            )
        )
    return records


class FrameKnowledgeGraph:
    """
    Relational knowledge graph organizing frames into hierarchy and reasoning propositions.
    """

    def __init__(self, records: List[FrameRecord]):
        self.records: Dict[str, FrameRecord] = {r.frame_id: r for r in records}
        self.children: Dict[str, List[str]] = {r.frame_id: [] for r in records}
        self.roots: List[str] = []

        for r in records:
            if r.parent_id != "NONE" and r.parent_id in self.children:
                self.children[r.parent_id].append(r.frame_id)
            elif r.parent_id == "NONE":
                self.roots.append(r.frame_id)

    def get_initial_evidence(self, frame_id: str) -> Tuple[float, float, float]:
        """
        Map frame attributes into evidence counts (S, F) and weight multiplier.
        Status:
            - 'completed': S=3.0, F=0.2 (strong success)
            - 'active':    S=1.0, F=1.0 (unresolved / equal chance)
            - 'pending':   S=0.1, F=2.0 (not started / delayed)
        Priority:
            - 'high':   weight = 1.5
            - 'medium': weight = 1.0
            - 'low':    weight = 0.7
        Returns (S, F, weight)
        """
        frame = self.records[frame_id]
        status_map = {
            "completed": (3.0, 0.2),
            "active": (1.0, 1.0),
            "pending": (0.1, 2.0),
        }
        s, f = status_map.get(frame.status, (1.0, 1.0))

        priority_map = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.7,
        }
        w = priority_map.get(frame.priority, 1.0)
        return s, f, w

    def to_pln_belief(self, frame_id: str, alpha_0: float = 1.0, beta_0: float = 1.0) -> PLNBelief:
        s, f, w = self.get_initial_evidence(frame_id)
        alpha = alpha_0 + s * w
        beta = beta_0 + f * w
        dist = BetaDistribution(alpha, beta)
        return PLNBelief.from_beta(
            dist,
            provenance=frozenset({f"Frame_{frame_id}_{self.records[frame_id].status}"}),
            label=f"Goal_{frame_id}",
        )
