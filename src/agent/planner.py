"""
PlannerAgent with LangGraph workflow for the Vision-Language Embodied Agent.

This module provides the main planning agent that orchestrates task execution
using a LangGraph StateGraph with plan, execute, verify, and adapt nodes.

Key components:
- AgentState: TypedDict for workflow state
- PlannerAgent: Main agent class with LangGraph workflow
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated
import logging

from src.agent.controller import ThorController, StepResult
from src.agent.navigator import NavigatorAgent
from src.planning.recovery import RecoveryAction, RecoveryResult, RecoveryStrategy
from src.planning.verifier import Verifier, VerificationResult
from src.planning.task_decomposer import TaskDecomposer, TaskDecomposition, Subgoal, EnvironmentObservation
from src.perception.visual_encoder import VisualEncoder
from src.perception.detector import ObjectDetector
from src.config.settings import Settings, default_settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State for the LangGraph workflow.

    Attributes:
        instruction: The original task instruction.
        decomposition: Current task decomposition.
        current_subgoal_idx: Index of current subgoal being executed.
        retry_count: Number of retries for current subgoal.
        replan_count: Number of global replans attempted.
        executed_actions: List of actions that have been executed.
        verification_results: Results of verification checks.
        status: Current status (planning, executing, verifying, adapting, done, failed).
        error: Error message if failed, None otherwise.
        result: Final result of task execution.
    """
    instruction: str
    decomposition: Optional[Dict[str, Any]]
    current_subgoal_idx: int
    retry_count: int
    replan_count: int
    executed_actions: List[str]
    verification_results: List[Dict[str, Any]]
    status: str
    error: Optional[str]
    result: Optional[Dict[str, Any]]


def _init_state(instruction: str) -> AgentState:
    """Create initial state for a new task."""
    return AgentState(
        instruction=instruction,
        decomposition=None,
        current_subgoal_idx=0,
        retry_count=0,
        replan_count=0,
        executed_actions=[],
        verification_results=[],
        status="planning",
        error=None,
        result=None
    )


class PlannerAgent:
    """
    Planning agent with LangGraph workflow for task execution.

    This agent uses a StateGraph with the following nodes:
    - plan: Decompose task into subgoals
    - execute: Execute the current subgoal
    - verify: Verify subgoal completion
    - adapt: Handle failures with recovery strategies

    Conditional edges from verify node:
    - continue: Move to next subgoal
    - retry: Retry current subgoal
    - replan: Decompose task again
    - done: Task completed successfully

    Example:
        >>> controller = ThorController(use_thor=False)
        >>> controller.reset("FloorPlan1")
        >>> agent = PlannerAgent(controller, use_llm_planner=False)
        >>> result = agent.execute_task("find the red mug")
    """

    def __init__(
        self,
        controller: ThorController,
        use_llm_planner: bool = True,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the PlannerAgent.

        Args:
            controller: ThorController for environment interaction.
            use_llm_planner: If True, use LLM for task decomposition.
                            If False, use vision-only template-based planning.
            settings: Settings instance. Uses default_settings if None.
        """
        self._controller = controller
        self._use_llm_planner = use_llm_planner
        self._settings = settings or default_settings

        # Initialize components
        self._decomposer = TaskDecomposer(
            use_llm=use_llm_planner,
            api_key=self._settings.llm.api_key,
            model=self._settings.llm.model,
            temperature=self._settings.llm.temperature,
            settings=self._settings
        )
        self._verifier = Verifier(controller, self._settings)
        self._recovery = RecoveryStrategy(
            max_retries=self._settings.planning.max_retries_per_subgoal,
            max_replans=self._settings.planning.max_global_replans,
            retry_rotation_degrees=self._settings.planning.retry_rotation_degrees,
            local_search_radius=self._settings.planning.local_search_radius
        )

        # Navigator for actual navigation (initialized lazily)
        self._navigator: Optional[NavigatorAgent] = None

        # Build the workflow graph
        self._graph = self._build_workflow()

        # Current state (for tracking without full LangGraph)
        self._state: Optional[AgentState] = None

    def _build_workflow(self) -> Any:
        """
        Build the LangGraph StateGraph workflow.

        Returns:
            Compiled StateGraph or None if LangGraph not available.
        """
        try:
            from langgraph.graph import StateGraph, END

            # Create the graph
            workflow = StateGraph(AgentState)

            # Add nodes
            workflow.add_node("plan", self._plan_node)
            workflow.add_node("execute", self._execute_node)
            workflow.add_node("verify", self._verify_node)
            workflow.add_node("adapt", self._adapt_node)

            # Add edges
            workflow.set_entry_point("plan")

            # plan -> execute
            workflow.add_edge("plan", "execute")

            # execute -> verify
            workflow.add_edge("execute", "verify")

            # verify -> conditional edges
            workflow.add_conditional_edges(
                "verify",
                self._should_continue,
                {
                    "continue": "execute",  # Move to next subgoal
                    "retry": "adapt",       # Retry current subgoal
                    "replan": "plan",       # Re-decompose task
                    "done": END             # Task complete
                }
            )

            # adapt -> execute (after recovery)
            workflow.add_edge("adapt", "execute")

            # Compile the graph
            return workflow.compile()

        except ImportError:
            logger.warning(
                "LangGraph not installed. Using simple execution loop."
            )
            return None

    def _plan_node(self, state: AgentState) -> AgentState:
        """
        Plan node: Decompose task into subgoals.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with decomposition.
        """
        instruction = state["instruction"]
        logger.info(f"Planning: decomposing task '{instruction}'")

        # Gather current environment observation
        environment = self._get_environment_observation()
        logger.info(f"Environment: {len(environment.visible_objects)} visible objects")

        # Decompose the task with environment context
        decomposition = self._decomposer.decompose(instruction, environment)

        # Update state
        state["decomposition"] = decomposition.model_dump()
        state["status"] = "executing"
        state["current_subgoal_idx"] = 0
        state["retry_count"] = 0

        logger.info(
            f"Plan created: {len(decomposition.subgoals)} subgoals, "
            f"~{decomposition.estimated_steps} steps"
        )

        return state

    def _get_environment_observation(self) -> EnvironmentObservation:
        """
        Gather current environment state for perception-aware planning.

        Returns:
            EnvironmentObservation with visible objects, position, etc.
        """
        try:
            # First, do a Pass action to get fresh observation from THOR
            self._controller.step("Pass")

            observation = self._controller.get_current_observation()
            state = self._controller.get_current_state()

            # Format visible objects
            visible_objects = []
            for obj in observation.visible_objects:
                visible_objects.append({
                    "name": obj.get("objectType", "Unknown"),
                    "distance": obj.get("distance", 0),
                    "position": obj.get("position", {"x": 0, "y": 0, "z": 0})
                })

            logger.info(f"Environment observation: {len(visible_objects)} visible objects")

            return EnvironmentObservation(
                visible_objects=visible_objects,
                agent_position=dict(state.position),
                held_object=state.held_object,
                agent_rotation=dict(state.rotation)
            )

        except Exception as e:
            logger.warning(f"Failed to get environment observation: {e}")
            return EnvironmentObservation()

    def _execute_node(self, state: AgentState) -> AgentState:
        """
        Execute node: Execute the current subgoal.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with execution results.
        """
        decomposition = state.get("decomposition")
        if not decomposition:
            state["status"] = "failed"
            state["error"] = "No decomposition available"
            return state

        subgoals = decomposition.get("subgoals", [])
        current_idx = state["current_subgoal_idx"]

        if current_idx >= len(subgoals):
            # All subgoals completed
            state["status"] = "done"
            return state

        # Get current subgoal
        subgoal = subgoals[current_idx]
        action = subgoal["action"]
        target = subgoal["target"]

        logger.info(f"Executing subgoal {current_idx + 1}/{len(subgoals)}: {action} -> {target}")

        # Execute the action
        action_result = self._execute_subgoal(action, target)

        # Update state
        state["executed_actions"].append(f"{action}:{target}")
        state["status"] = "verifying"

        if not action_result.get("success", False):
            state["error"] = action_result.get("error", "Unknown error")

        return state

    def _verify_node(self, state: AgentState) -> AgentState:
        """
        Verify node: Check if subgoal was completed successfully.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with verification results.
        """
        decomposition = state.get("decomposition")
        if not decomposition:
            state["status"] = "failed"
            return state

        subgoals = decomposition.get("subgoals", [])
        current_idx = state["current_subgoal_idx"]

        if current_idx >= len(subgoals):
            state["status"] = "done"
            return state

        subgoal = subgoals[current_idx]
        action = subgoal["action"]
        target = subgoal["target"]

        logger.info(f"Verifying subgoal: {action} -> {target}")

        # Perform verification based on action type
        verification_result = self._verify_subgoal(action, target)

        # Record verification result
        state["verification_results"].append({
            "subgoal_idx": current_idx,
            "action": action,
            "target": target,
            "success": verification_result.success,
            "message": verification_result.message
        })

        if verification_result.success:
            # Move to next subgoal
            state["current_subgoal_idx"] = current_idx + 1
            state["retry_count"] = 0
            state["error"] = None
            logger.info(f"Subgoal verified: moving to next subgoal")

            # Check if all subgoals are complete
            if state["current_subgoal_idx"] >= len(subgoals):
                state["status"] = "done"
                logger.info("All subgoals completed successfully")
        else:
            # Record failure
            state["error"] = verification_result.message
            logger.warning(f"Subgoal verification failed: {verification_result.message}")

        return state

    def _adapt_node(self, state: AgentState) -> AgentState:
        """
        Adapt node: Handle failures with recovery strategies.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with recovery actions applied.
        """
        retry_count = state["retry_count"]
        replan_count = state["replan_count"]
        error = state.get("error", "unknown")

        logger.info(f"Adapting: retry_count={retry_count}, replan_count={replan_count}")

        # Select recovery action
        recovery_action = self._recovery.select_strategy(
            retry_count=retry_count,
            replan_count=replan_count,
            failure_type="unknown"
        )

        logger.info(f"Selected recovery action: {recovery_action.value}")

        if recovery_action == RecoveryAction.RETRY:
            # Increment retry count
            state["retry_count"] = retry_count + 1

            # Execute retry actions (rotate to get different view)
            self._execute_retry_actions()

        elif recovery_action == RecoveryAction.LOCAL_SEARCH:
            # Execute local search
            self._execute_local_search()

            # Increment retry count for tracking
            state["retry_count"] = retry_count + 1

        elif recovery_action == RecoveryAction.REPLAN:
            # Increment replan count
            state["replan_count"] = replan_count + 1
            state["retry_count"] = 0
            state["status"] = "planning"

        elif recovery_action == RecoveryAction.ABORT:
            # Give up
            state["status"] = "failed"
            state["error"] = f"Task failed after {retry_count} retries and {replan_count} replans: {error}"

        return state

    def _should_continue(self, state: AgentState) -> str:
        """
        Determine the next step after verification.

        Args:
            state: Current workflow state.

        Returns:
            String indicating next action: continue, retry, replan, or done.
        """
        # Check if all subgoals are complete
        decomposition = state.get("decomposition")
        if not decomposition:
            return "done"

        subgoals = decomposition.get("subgoals", [])
        current_idx = state["current_subgoal_idx"]

        # If we've completed all subgoals
        if current_idx >= len(subgoals):
            return "done"

        # Check last verification result
        verification_results = state.get("verification_results", [])
        if verification_results:
            last_result = verification_results[-1]
            if not last_result.get("success", False):
                # Subgoal failed - decide recovery action
                retry_count = state["retry_count"]
                replan_count = state["replan_count"]

                recovery_action = self._recovery.select_strategy(
                    retry_count=retry_count,
                    replan_count=replan_count,
                    failure_type="unknown"
                )

                if recovery_action == RecoveryAction.RETRY:
                    return "retry"
                elif recovery_action == RecoveryAction.LOCAL_SEARCH:
                    return "retry"  # Treat as retry for simplicity
                elif recovery_action == RecoveryAction.REPLAN:
                    return "replan"
                else:
                    return "done"  # Abort

        # Success - continue to next subgoal
        return "continue"

    def execute_task(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a task instruction.

        This is the main entry point for task execution. It either uses
        the LangGraph workflow or a simple execution loop.

        Args:
            instruction: The task instruction to execute.

        Returns:
            Dict with success status, result, and error (if any).
        """
        logger.info(f"Starting task execution: '{instruction}'")

        # Reset state for new task
        self._state = None
        self._navigator = None  # Reset navigator to get fresh observation

        # Initialize state
        self._state = _init_state(instruction)

        # Try using LangGraph if available
        if self._graph is not None:
            return self._execute_with_langgraph()

        # Fall back to simple execution loop
        return self._execute_simple_loop()

    def _execute_with_langgraph(self) -> Dict[str, Any]:
        """
        Execute task using LangGraph workflow.

        Returns:
            Dict with execution results.
        """
        try:
            # Run the graph
            final_state = self._graph.invoke(self._state)

            # Extract result
            success = final_state.get("status") == "done"
            error = final_state.get("error")

            return {
                "success": success,
                "executed_actions": final_state.get("executed_actions", []),
                "verification_results": final_state.get("verification_results", []),
                "decomposition": final_state.get("decomposition"),
                "error": error
            }

        except Exception as e:
            logger.error(f"LangGraph execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "executed_actions": [],
                "verification_results": [],
                "decomposition": None
            }

    def _execute_simple_loop(self) -> Dict[str, Any]:
        """
        Execute task using a simple loop (when LangGraph not available).

        Returns:
            Dict with execution results.
        """
        max_iterations = self._settings.planning.max_subgoals * (
            self._settings.planning.max_retries_per_subgoal + 1
        ) * (self._settings.planning.max_global_replans + 1)

        for iteration in range(max_iterations):
            status = self._state.get("status")

            if status == "planning":
                self._state = self._plan_node(self._state)

            elif status == "executing":
                self._state = self._execute_node(self._state)

            elif status == "verifying":
                self._state = self._verify_node(self._state)

                # Check what to do next
                next_action = self._should_continue(self._state)

                if next_action == "done":
                    self._state["status"] = "done"
                elif next_action == "retry" or next_action == "replan":
                    self._state = self._adapt_node(self._state)

                # Continue loop to check final status
                continue

            elif status == "done":
                break

            elif status == "failed":
                break

        # Extract result
        success = self._state.get("status") == "done"
        error = self._state.get("error") if not success else None

        return {
            "success": success,
            "executed_actions": self._state.get("executed_actions", []),
            "verification_results": self._state.get("verification_results", []),
            "decomposition": self._state.get("decomposition"),
            "error": error
        }

    def _execute_subgoal(self, action: str, target: str) -> Dict[str, Any]:
        """
        Execute a single subgoal.

        Args:
            action: Action to perform.
            target: Target object or location.

        Returns:
            Dict with success status and error (if any).
        """
        try:
            if action == "navigate":
                # Use NavigatorAgent for actual navigation
                if self._navigator is None:
                    # Initialize navigator lazily
                    visual_encoder = VisualEncoder(
                        model_name=self._settings.perception.clip_model,
                        pretrained=self._settings.perception.clip_pretrained
                    )
                    detector = ObjectDetector(
                        model_name=self._settings.perception.yolo_model
                    )
                    self._navigator = NavigatorAgent(
                        controller=self._controller,
                        visual_encoder=visual_encoder,
                        detector=detector,
                        settings=self._settings
                    )

                # Navigate to target
                logger.info(f"Navigating to: {target}")
                nav_result = self._navigator.navigate_to_target(target)

                if nav_result.get("success"):
                    logger.info(f"Navigation successful: reached {target}")
                    return {"success": True, "error": None}
                else:
                    error = nav_result.get("error", "navigation failed")
                    logger.warning(f"Navigation failed: {error}")
                    return {"success": False, "error": error}

            elif action == "pickup":
                # Find object and pick it up
                observation = self._controller.get_current_observation()
                matched_object = self._find_object_in_view(target, observation.visible_objects)

                if matched_object is None:
                    return {"success": False, "error": f"Cannot find '{target}' in view"}

                result = self._controller.step("PickupObject", objectId=matched_object["objectId"])
                return {"success": result.success, "error": result.error}

            elif action == "put":
                result = self._controller.step("PutObject")
                return {"success": result.success, "error": result.error}

            elif action == "open":
                observation = self._controller.get_current_observation()
                matched_object = self._find_object_in_view(target, observation.visible_objects)

                if matched_object is None:
                    return {"success": False, "error": f"Cannot find '{target}' in view"}

                result = self._controller.step("OpenObject", objectId=matched_object["objectId"])
                return {"success": result.success, "error": result.error}

            elif action == "close":
                observation = self._controller.get_current_observation()
                matched_object = self._find_object_in_view(target, observation.visible_objects)

                if matched_object is None:
                    return {"success": False, "error": f"Cannot find '{target}' in view"}

                result = self._controller.step("CloseObject", objectId=matched_object["objectId"])
                return {"success": result.success, "error": result.error}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Error executing subgoal: {e}")
            return {"success": False, "error": str(e)}

    def _find_object_in_view(self, target: str, visible_objects: List[Dict]) -> Optional[Dict]:
        """
        Find a matching object in visible objects.

        Args:
            target: Target object name.
            visible_objects: List of visible objects from observation.

        Returns:
            Matching object dict or None.
        """
        target_lower = target.lower()
        for obj in visible_objects:
            obj_type = obj.get("objectType", "").lower()
            # Match if target is in object type or object type is in target
            if target_lower in obj_type or obj_type in target_lower:
                return obj
        return None

    def _verify_subgoal(self, action: str, target: str) -> VerificationResult:
        """
        Verify a subgoal was completed.

        Args:
            action: Action that was performed.
            target: Target object or location.

        Returns:
            VerificationResult with success status.
        """
        if action == "navigate":
            # For navigation, we verify by checking position
            # In a full implementation, this would check distance to target
            current_state = self._controller.get_current_state()
            return VerificationResult(
                success=True,
                message=f"Navigation verified at position {current_state.position}"
            )

        elif action == "pickup":
            return self._verifier.verify_pickup(target)

        elif action == "put":
            return self._verifier.verify_put(target, "receptacle")

        elif action == "open":
            return self._verifier.verify_open(target)

        elif action == "close":
            return self._verifier.verify_close(target)

        else:
            return VerificationResult(
                success=False,
                message=f"Cannot verify unknown action: {action}"
            )

    def _execute_retry_actions(self) -> None:
        """Execute actions to prepare for retry (e.g., rotation)."""
        actions = self._recovery.get_retry_actions()
        for action in actions:
            self._controller.step(action)

    def _execute_local_search(self) -> None:
        """Execute local search pattern."""
        actions = self._recovery.get_local_search_actions()
        for action_name, kwargs in actions:
            self._controller.step(action_name, **kwargs)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._state = None
        logger.info("PlannerAgent reset")

    def get_state(self) -> Optional[AgentState]:
        """
        Get the current workflow state.

        Returns:
            Current AgentState or None if not executing.
        """
        return self._state
