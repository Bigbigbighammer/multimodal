---
name: Vision-Language Embodied Agent
date: 2026-04-22
status: approved
---

# Vision-Language Embodied Agent Design Spec

## Overview

Build a "Vision + Language" embodied navigation/interaction agent in AI2-THOR, using LangChain 1.0 + LangGraph for agent orchestration. The architecture follows Claude Code's Plan→Execute→Verify→Adapt pattern with three-layer hierarchical planning.

## Constraints

- **GPU**: RTX 4060 Laptop, 8GB VRAM
- **Python**: 3.11.9
- **OS**: Windows 11

VRAM budget:
- CLIP ViT-B/32: ~0.6GB
- YOLOv8-nano: ~0.1GB
- AI2-THOR rendering: ~1.5GB
- System/buffer: remaining

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Simulation | AI2-THOR | 5.x |
| Agent Framework | LangChain + LangGraph | 0.3.x / 0.2.x |
| LLM | OpenAI GPT-4o-mini | API |
| Visual Encoding | CLIP (ViT-B/32) | OpenAI |
| Object Detection | YOLOv8-nano | ultralytics |
| Map | Custom 2D topological map | - |
| Visualization | matplotlib + PIL | - |

## Architecture: Three-Layer Planning

### Layer 1: Controller (Bottom)

Raw AI2-THOR interface wrapper.

**Action space:**
- Movement: MoveAhead, MoveBack, MoveLeft, MoveRight
- Rotation: RotateLeft, RotateRight
- Camera: LookUp, LookDown
- Interaction: PickupObject, PutObject, OpenObject, CloseObject, DropHandObject

**ThorObservation contains:**
- RGB image (first-person view)
- Depth image
- Instance segmentation mask
- Agent state (position, rotation, held_object)
- Visible objects metadata

```python
class ThorController:
    def step(action: str, **kwargs) -> ThorObservation
    def reset(scene_name: str, initial_position: dict) -> ThorObservation
    def get_reachable_positions() -> List[dict]
    def get_current_state() -> AgentState
```

### Layer 2: NavigatorAgent (Middle)

Sub-goal level path planning and execution.

```python
class NavigatorAgent:
    def navigate_to(target_description: str) -> NavigationResult
    def build_spatial_map() -> TopologicalMap
    def find_object(description: str) -> Optional[ObjectInfo]
```

**Perception pipeline:**
1. RGB Image → YOLO → object bounding boxes
2. Crop each detected object → CLIP image encoder → per-object features
3. Target description → CLIP text encoder → text feature
4. Cosine similarity → best matching object
5. A* path planning on topological map → execute movement steps
6. Re-perceive at each step

### Layer 3: PlannerAgent (Top)

Task decomposition and supervision, following Claude Code's design:

```python
class PlannerAgent:
    def execute_task(instruction: str) -> TaskResult
```

**Core loop:** Plan → Execute → Verify → Adapt

1. **Plan**: LLM decomposes instruction into ordered subgoals
2. **Execute**: Delegate subgoal to Navigator/Controller
3. **Verify**: Check if subgoal completed (distance threshold, object in hand, etc.)
4. **Adapt**: On failure, choose recovery strategy

**Recovery strategies (escalating):**
1. Single-step failure: rotate 90° and retry
2. Sub-goal failure: re-search target location (local replan)
3. 3 consecutive failures: global replan via LLM
4. Unrecoverable: terminate with failure report

## LangGraph Workflow

```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    instruction: str
    subgoals: list
    current_subgoal_idx: int
    observations: list
    action_history: list
    spatial_map: dict
    retry_count: int
    result: str

workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_subgoals)
workflow.add_node("execute", execute_subgoal)
workflow.add_node("verify", verify_result)
workflow.add_node("adapt", replan)

workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "verify")
workflow.add_conditional_edges("verify", should_continue, {
    "continue": "execute",   # next subgoal
    "retry": "execute",      # retry current subgoal
    "replan": "adapt",       # replan via LLM
    "done": END
})
workflow.add_edge("adapt", "execute")
```

LangChain 1.0 components used inside graph nodes:
- `ChatOpenAI` for LLM calls
- `ChatPromptTemplate` for prompt management
- Output parsers for structured LLM output

## Memory System

### Topological Map

```python
class TopologicalMap:
    nodes: Dict[Position, MapNode]
    edges: Set[Tuple[Position, Position]]

class MapNode:
    position: Tuple[float, float]
    objects_seen: List[str]
    image_features: torch.Tensor
    visited_count: int
    last_visited: float
```

- Use `get_reachable_positions()` from AI2-THOR for all reachable points
- Record observed objects (YOLO results) at each visited position
- Mark explored areas via `visited_count`
- When searching for a target, prioritize nodes where target was seen but not reached

### Working Memory

```python
class WorkingMemory:
    # Task level
    original_instruction: str
    current_plan: List[SubGoal]
    plan_revision_count: int

    # Execution level
    action_history: List[ActionRecord]
    held_object: Optional[str]

    # Perception level
    current_visible_objects: List[ObjectInfo]
    target_object_location: Optional[Position]

    def get_context_for_llm() -> str  # Summarized context for LLM
    def should_summarize() -> bool     # Trigger compression when history exceeds threshold
```

Context compression strategy: keep last 10 action steps + current goal + scene description. Similar to Claude Code's context management.

## Task Types

### ObjectNav (Object Goal Navigation)

- **Input**: Target category (e.g., "chair")
- **Success**: Agent within 1.0m of any instance of target category, with target visible
- **Metric**: SPL = (1/N) * Σ(S_i * l_i / max(p_i, l_i))

### VLN (Vision-Language Navigation)

- **Input**: Natural language instruction (e.g., "走到红色的椅子旁边")
- **Success**: Agent reaches the described target
- **Additional**: Color/attribute matching via CLIP

### Simple Interaction

- **Input**: Multi-step instruction (e.g., "走到桌子旁，拿起苹果")
- **Success**: All subgoals completed
- **Actions**: PickupObject, OpenObject, CloseObject

## Comparison Experiment

| Dimension | Vision-Only Nav | Vision+Language Nav |
|-----------|-----------------|---------------------|
| Input | Category name | Natural language |
| Planner | No LLM, direct CLIP match | LLM decompose → CLIP match |
| Path strategy | Greedy (move to best match) | A* plan + verify loop |
| Test set | Same scenes | Same scenes |
| Metrics | SPL, success rate, steps | SPL, success rate, steps |

## Evaluation System

### Metrics

```python
@dataclass
class EpisodeMetrics:
    success: bool
    spl: float
    total_steps: int
    total_distance: float
    planning_efficiency: float  # completed_subgoals / total_subgoals
```

### Visualization

- Top-down floor plan with trajectory path
- Key frames at decision points
- Output per episode: `results/episode_{id}/` with trajectory image, keyframe GIF, metrics JSON

### Evaluation Pipeline

```python
# evaluate.py
runner = EvaluationRunner(
    scenes=["FloorPlan1", ..., "FloorPlan10"],
    tasks_per_scene=3,
    max_steps_per_episode=200
)
results = runner.run_all()
report = results.generate_report()  # success_rate, avg_spl, comparison tables
```

## Project Structure

```
multimodel/
├── src/
│   ├── agent/
│   │   ├── planner.py          # PlannerAgent - high-level decomposition
│   │   ├── navigator.py        # NavigatorAgent - mid-level path planning
│   │   └── controller.py       # Controller - THOR interface
│   ├── perception/
│   │   ├── visual_encoder.py   # CLIP feature extraction
│   │   ├── detector.py         # YOLO object detection
│   │   └── scene_graph.py      # Scene understanding
│   ├── memory/
│   │   ├── working_memory.py   # Short-term task state
│   │   ├── spatial_map.py      # 2D topological map
│   │   └── episode_history.py  # Episode trajectory recording
│   ├── planning/
│   │   ├── task_decomposer.py  # Long task → subgoal decomposition
│   │   ├── recovery.py         # Error recovery strategies
│   │   └── verifier.py         # Subgoal completion verification
│   ├── evaluation/
│   │   ├── metrics.py          # SPL, Success Rate
│   │   ├── runner.py           # Batch episode runner
│   │   └── visualizer.py       # Trajectory visualization
│   └── config/
│       └── settings.py         # Configuration
├── tasks/
│   ├── objectnav.yaml
│   ├── vln.yaml
│   └── interaction.yaml
├── results/                    # Evaluation outputs
├── main.py                     # Entry point
├── evaluate.py                 # Evaluation entry point
└── requirements.txt
```

## Dependencies

```
ai2thor>=5.0.0
langchain>=0.3.0
langgraph>=0.2.0
langchain-openai>=0.2.0
open-clip-torch>=2.24.0
ultralytics>=8.1.0
torch>=2.1.0
torchvision>=0.16.0
matplotlib>=3.8.0
Pillow>=10.0.0
pyyaml>=6.0
numpy>=1.24.0
tqdm>=4.66.0
```
