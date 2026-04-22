"""Episode history recording."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime


@dataclass
class StepRecord:
    """Record of a single step."""
    step: int
    action: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    success: bool
    visible_objects: List[str] = field(default_factory=list)


@dataclass
class EpisodeHistory:
    """History of a complete episode."""
    episode_id: int
    task_type: str
    scene: str
    instruction: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())

    steps: List[StepRecord] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)

    success: bool = False
    total_steps: int = 0
    end_time: Optional[str] = None

    def add_step(
        self,
        action: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        success: bool,
        visible_objects: List[str] = None,
    ):
        """Add a step record."""
        step = StepRecord(
            step=len(self.steps),
            action=action,
            position=position,
            rotation=rotation,
            success=success,
            visible_objects=visible_objects or [],
        )
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def finalize(self, success: bool):
        """Mark episode as complete."""
        self.success = success
        self.end_time = datetime.now().isoformat()

    def get_positions(self) -> List[Tuple[float, float, float]]:
        """Get all positions for trajectory."""
        return [s.position for s in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "task_type": self.task_type,
            "scene": self.scene,
            "instruction": self.instruction,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "success": self.success,
            "total_steps": self.total_steps,
            "subgoals": self.subgoals,
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "position": list(s.position),
                    "rotation": list(s.rotation),
                    "success": s.success,
                    "visible_objects": s.visible_objects,
                }
                for s in self.steps
            ],
        }

    def save(self, output_dir: str):
        """Save to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"episode_{self.episode_id}.json"
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return file_path
