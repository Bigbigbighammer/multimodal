"""
Configuration settings for the Vision-Language Embodied Agent.

This module defines all configuration parameters for AI2-THOR simulation,
perception models, navigation, planning, LLM integration, and memory management.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThorSettings:
    """AI2-THOR simulation settings."""
    grid_size: float = 0.25  # MoveAhead distance in meters
    render_depth: bool = True
    render_instance_segmentation: bool = True
    width: int = 640
    height: int = 480
    visibility_distance: float = 1.5  # Max distance for object visibility

    # Controller settings
    rotate_step_degrees: float = 90.0
    horizon_step_degrees: float = 30.0
    snap_to_grid: bool = True


@dataclass
class PerceptionSettings:
    """Perception model settings for CLIP and YOLO."""
    # CLIP settings
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_match_threshold: float = 0.25  # Cosine similarity threshold for matching

    # YOLO settings
    yolo_model: str = "yolov8n.pt"  # nano model for efficiency
    yolo_confidence: float = 0.3  # Minimum confidence for detections
    yolo_iou_threshold: float = 0.5

    # Feature caching
    cache_features: bool = True


@dataclass
class NavigationSettings:
    """Navigation and path planning settings."""
    success_distance: float = 1.0  # Distance in meters to consider goal reached
    max_steps_per_episode: int = 200

    # Exploration settings
    exploration_frontier_weight: float = 2.0  # Weight for unexplored areas
    max_exploration_steps: int = 50

    # Path planning
    path_replan_threshold: float = 0.5  # Replan if off-path by this distance

    # Movement
    move_distance: float = 0.25  # Default MoveAhead distance
    collision_threshold: float = 0.1  # Minimum distance to objects


@dataclass
class PlanningSettings:
    """High-level planning and task decomposition settings."""
    max_retries_per_subgoal: int = 2
    max_global_replans: int = 1

    # Recovery behavior
    retry_rotation_degrees: float = 90.0  # Rotate this much before retry
    local_search_radius: float = 2.0  # Radius for local re-search

    # Verification
    verification_wait_steps: int = 2  # Steps to wait before verifying

    # Subgoal settings
    max_subgoals: int = 5  # Maximum subgoals per task (keep simple)


@dataclass
class LLMSettings:
    """LLM integration settings."""
    model: str = "gpt-5.2"
    temperature: float = 0.1
    max_tokens: int = 1024

    # API settings - loaded from environment variable OPENAI_API_KEY
    api_key: str = ""  # Set via OPENAI_API_KEY env var or config file
    api_base: Optional[str] = None  # Custom API endpoint if needed
    request_timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0


@dataclass
class MemorySettings:
    """Memory and context management settings."""
    working_memory_max_steps: int = 10  # Action history kept before summarization

    # Spatial map settings
    map_edge_distance_threshold: float = 0.30  # Edge connection threshold
    map_max_nodes: int = 1000  # Maximum nodes in spatial map

    # Episode history
    max_episode_history: int = 100  # Episodes to keep in history

    # Context compression
    context_compression_threshold: int = 20  # Actions before compression


@dataclass
class EvaluationSettings:
    """Evaluation and visualization settings."""
    # Metrics
    num_episodes_per_scene: int = 3
    success_distance_threshold: float = 1.0
    success_angle_threshold: float = 30.0  # Degrees

    # Visualization
    save_trajectory_images: bool = True
    save_keyframe_gifs: bool = True
    visualization_fps: int = 5

    # Output
    results_dir: str = "results"
    save_episode_videos: bool = False


@dataclass
class Settings:
    """
    Main configuration class containing all settings for the embodied agent.

    Settings are organized into logical groups matching the system architecture:
    - thor: AI2-THOR simulation parameters
    - perception: CLIP and YOLO model settings
    - navigation: Path planning and movement parameters
    - planning: Task decomposition and recovery settings
    - llm: OpenAI API configuration
    - memory: Working memory and spatial map settings
    - evaluation: Metrics and visualization options
    """
    thor: ThorSettings = field(default_factory=ThorSettings)
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)
    navigation: NavigationSettings = field(default_factory=NavigationSettings)
    planning: PlanningSettings = field(default_factory=PlanningSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    # Global settings
    seed: int = 42  # Random seed for reproducibility
    debug: bool = False
    log_level: str = "INFO"

    # Mode flags
    use_llm_planner: bool = True  # If False, uses vision-only navigation
    headless: bool = False  # Run AI2-THOR without rendering window

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create Settings instance from environment variables.

        Environment variables should be prefixed with AGENT_ and use
        double underscores for nested settings. Examples:
        - AGENT_LLM__API_KEY
        - AGENT_PERCEPTION__CLIP_MODEL
        - AGENT_THOR__GRID_SIZE
        - AGENT_DEBUG

        Returns:
            Settings instance with values from environment variables
        """
        settings = cls()

        # Load LLM API key
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            settings.llm.api_key = api_key

        # Load optional API base
        api_base = os.environ.get("OPENAI_API_BASE", "")
        if api_base:
            settings.llm.api_base = api_base

        # Load simple boolean/integer settings from environment
        if os.environ.get("AGENT_DEBUG", "").lower() in ("true", "1", "yes"):
            settings.debug = True

        if os.environ.get("AGENT_HEADLESS", "").lower() in ("true", "1", "yes"):
            settings.headless = True

        if os.environ.get("AGENT_USE_LLM_PLANNER", "").lower() in ("false", "0", "no"):
            settings.use_llm_planner = False

        seed_str = os.environ.get("AGENT_SEED")
        if seed_str:
            settings.seed = int(seed_str)

        log_level = os.environ.get("AGENT_LOG_LEVEL")
        if log_level:
            settings.log_level = log_level

        # Load nested settings from environment
        # Format: AGENT_<SECTION>__<PARAM> (e.g., AGENT_THOR__GRID_SIZE)
        _load_nested_env(settings.thor, "THOR")
        _load_nested_env(settings.perception, "PERCEPTION")
        _load_nested_env(settings.navigation, "NAVIGATION")
        _load_nested_env(settings.planning, "PLANNING")
        _load_nested_env(settings.llm, "LLM")
        _load_nested_env(settings.memory, "MEMORY")
        _load_nested_env(settings.evaluation, "EVALUATION")

        return settings

    def to_dict(self) -> dict:
        """Convert settings to a flat dictionary for serialization."""
        result = {
            "seed": self.seed,
            "debug": self.debug,
            "log_level": self.log_level,
            "use_llm_planner": self.use_llm_planner,
            "headless": self.headless,
        }

        for section_name in ["thor", "perception", "navigation", "planning",
                            "llm", "memory", "evaluation"]:
            section = getattr(self, section_name)
            for field_name in section.__dataclass_fields__:
                key = f"{section_name}.{field_name}"
                result[key] = getattr(section, field_name)

        return result


def _load_nested_env(obj: object, section: str) -> None:
    """
    Load environment variables into a nested settings object.

    Args:
        obj: The settings dataclass instance to update
        section: The section name (e.g., "THOR", "PERCEPTION")
    """
    for field_name in obj.__dataclass_fields__:
        env_key = f"AGENT_{section}__{field_name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is None:
            continue

        field_type = obj.__dataclass_fields__[field_name].type
        try:
            if field_type == bool:
                setattr(obj, field_name, env_value.lower() in ("true", "1", "yes"))
            elif field_type == int:
                setattr(obj, field_name, int(env_value))
            elif field_type == float:
                setattr(obj, field_name, float(env_value))
            elif field_type == str:
                setattr(obj, field_name, env_value)
            elif field_type == Optional[str]:
                setattr(obj, field_name, env_value if env_value else None)
            else:
                setattr(obj, field_name, env_value)
        except (ValueError, TypeError):
            pass  # Keep default value on conversion error


# Default settings instance
default_settings = Settings()
