# Plan: Perception-Aware Task Decomposition

## Problem Statement

The current system decomposes tasks without considering the current environment state. LLM generates plans blindly, resulting in:
- Over-planning (11 subgoals for "find apple" when apple is visible at 1.2m)
- Unnecessary exploration steps
- No utilization of already-visible objects

## Goal

Implement perception-aware task decomposition where LLM receives current environment state (visible objects, distances, held objects) before making planning decisions.

## Architecture Change

### Current Flow
```
用户输入 → LLM分解（无环境信息）→ 执行 → 验证
```

### New Flow
```
用户输入 → 感知环境 → LLM决策（结合环境信息）→ 执行 → 验证 → 失败重感知
```

## Implementation Tasks

### Task 1: Add EnvironmentObservation dataclass
**File:** `src/planning/task_decomposer.py`

Add a new dataclass to represent environment state:
```python
@dataclass
class EnvironmentObservation:
    """Current environment state for planning."""
    visible_objects: List[Dict[str, Any]]  # [{name, distance, position}]
    agent_position: Dict[str, float]
    held_object: Optional[str]
    agent_rotation: Dict[str, float]
```

### Task 2: Update TaskDecomposer.decompose() to accept environment
**File:** `src/planning/task_decomposer.py`

Change signature:
```python
def decompose(
    self,
    instruction: str,
    environment: Optional[EnvironmentObservation] = None
) -> TaskDecomposition:
```

### Task 3: Update LLM prompt to include environment
**File:** `src/planning/task_decomposer.py`

New prompt template:
```python
DECOMPOSITION_PROMPT = """You are an embodied AI agent in a home environment.

## Current Environment State
{environment_section}

## Task
{task}

## Rules
1. If target object is already visible and close (< 2m), just navigate directly
2. Keep plans MINIMAL - no unnecessary exploration
3. Use visible objects information to make smart decisions

Return JSON with subgoals...
"""

def _format_environment(self, env: EnvironmentObservation) -> str:
    """Format environment for LLM prompt."""
    lines = ["Visible Objects:"]
    for obj in env.visible_objects[:10]:  # Limit to 10
        lines.append(f"  - {obj['name']} ({obj['distance']:.1f}m)")
    if env.held_object:
        lines.append(f"Holding: {env.held_object}")
    return "\n".join(lines)
```

### Task 4: Update PlannerAgent to gather environment before decomposition
**File:** `src/agent/planner.py`

Add method to gather environment:
```python
def _get_environment_observation(self) -> EnvironmentObservation:
    """Gather current environment state."""
    obs = self._controller.get_current_observation()
    state = self._controller.get_current_state()
    
    return EnvironmentObservation(
        visible_objects=[
            {"name": obj["objectType"], "distance": obj.get("distance", 0)}
            for obj in obs.visible_objects
        ],
        agent_position=state.position,
        held_object=state.held_object,
        agent_rotation=state.rotation
    )
```

Update `_plan_node`:
```python
def _plan_node(self, state: AgentState) -> AgentState:
    instruction = state["instruction"]
    
    # Gather environment
    environment = self._get_environment_observation()
    logger.info(f"Environment: {len(environment.visible_objects)} visible objects")
    
    # Decompose with environment
    decomposition = self._decomposer.decompose(instruction, environment)
    
    # ... rest unchanged
```

### Task 5: Update template-based decomposition to use environment
**File:** `src/planning/task_decomposer.py`

Check if target is already visible:
```python
def _decompose_with_templates(
    self,
    instruction: str,
    environment: Optional[EnvironmentObservation] = None
) -> TaskDecomposition:
    # Check if target is visible
    if environment:
        target = self._extract_target(instruction, "target_object")
        for obj in environment.visible_objects:
            if target.lower() in obj["name"].lower():
                if obj["distance"] < 2.0:
                    # Target is close, simple plan
                    return TaskDecomposition(
                        task=instruction,
                        subgoals=[Subgoal(
                            id="subgoal_1",
                            action="navigate",
                            target=obj["name"],
                            description=f"Target {obj['name']} visible at {obj['distance']:.1f}m",
                            dependencies=[]
                        )],
                        reasoning=f"Target already visible at {obj['distance']:.1f}m",
                        estimated_steps=2
                    )
    # ... rest unchanged
```

### Task 6: Add tests for perception-aware decomposition
**File:** `tests/test_task_decomposer.py`

Add tests:
- Test decomposition with visible target close by
- Test decomposition with no visible target
- Test environment observation gathering

## Files to Modify

1. `src/planning/task_decomposer.py` - Add EnvironmentObservation, update decompose()
2. `src/agent/planner.py` - Add _get_environment_observation(), update _plan_node()
3. `tests/test_task_decomposer.py` - Add new tests

## Success Criteria

1. When apple is visible at 1.2m, "find apple" generates 1 subgoal (not 11)
2. LLM prompt includes visible objects and distances
3. Template decomposition checks for visible targets first
4. All existing tests pass
5. New tests for perception-aware behavior pass
