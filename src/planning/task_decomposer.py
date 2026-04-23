"""
Task decomposition module for the Vision-Language Embodied Agent.

This module uses LangChain with LLM to decompose high-level task
instructions into executable subgoals.

Key components:
- EnvironmentObservation: Dataclass for current environment state
- TaskDecomposition: Pydantic model for structured decomposition output
- Subgoal: Pydantic model for individual subgoals
- TaskDecomposer: Class that performs task decomposition using LLM
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import json

from pydantic import BaseModel, Field

from src.config.settings import Settings, default_settings

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentObservation:
    """
    Current environment state for perception-aware planning.

    Attributes:
        visible_objects: List of visible objects with name, distance, position.
        agent_position: Current agent position {x, y, z}.
        held_object: Object currently held by agent, if any.
        agent_rotation: Current agent rotation {x, y, z}.
    """
    visible_objects: List[Dict[str, Any]] = field(default_factory=list)
    agent_position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    held_object: Optional[str] = None
    agent_rotation: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})

    def has_visible_object(self, name: str, max_distance: float = 2.0) -> Optional[Dict[str, Any]]:
        """
        Check if an object with given name is visible and within distance.

        Args:
            name: Object name to search for.
            max_distance: Maximum distance threshold.

        Returns:
            Matching object dict if found, None otherwise.
        """
        name_lower = name.lower()
        for obj in self.visible_objects:
            obj_name = obj.get("name", obj.get("objectType", "")).lower()
            if name_lower in obj_name or obj_name in name_lower:
                distance = obj.get("distance", float('inf'))
                if distance <= max_distance:
                    return obj
        return None

    def to_prompt_string(self) -> str:
        """
        Format environment for LLM prompt.

        Returns:
            Human-readable string describing environment.
        """
        lines = []

        if self.visible_objects:
            lines.append("Visible Objects:")
            # Sort by distance
            sorted_objects = sorted(
                self.visible_objects,
                key=lambda x: x.get("distance", float('inf'))
            )
            for obj in sorted_objects[:15]:  # Limit to 15 for prompt
                name = obj.get("name", obj.get("objectType", "Unknown"))
                distance = obj.get("distance", 0)
                lines.append(f"  - {name} ({distance:.1f}m)")
            if len(self.visible_objects) > 15:
                lines.append(f"  ... and {len(self.visible_objects) - 15} more")
        else:
            lines.append("No visible objects.")

        if self.held_object:
            lines.append(f"Holding: {self.held_object}")

        lines.append(f"Position: ({self.agent_position['x']:.1f}, {self.agent_position['z']:.1f})")

        return "\n".join(lines)


class Subgoal(BaseModel):
    """
    A single subgoal in a task decomposition.

    Attributes:
        id: Unique identifier for the subgoal (e.g., "subgoal_1").
        action: The action to perform (e.g., "navigate", "pickup", "open").
        target: The target object or location for the action.
        description: Human-readable description of the subgoal.
        dependencies: List of subgoal IDs that must complete first.
    """
    id: str = Field(..., description="Unique identifier for the subgoal")
    action: str = Field(..., description="The action to perform")
    target: str = Field(..., description="The target object or location")
    description: str = Field(..., description="Human-readable description")
    dependencies: List[str] = Field(default_factory=list, description="Required prior subgoals")


class TaskDecomposition(BaseModel):
    """
    Structured output for task decomposition.

    Attributes:
        task: The original task instruction.
        subgoals: List of subgoals to execute.
        reasoning: Explanation of the decomposition strategy.
        estimated_steps: Estimated number of steps to complete.
    """
    task: str = Field(..., description="The original task instruction")
    subgoals: List[Subgoal] = Field(..., description="List of subgoals to execute")
    reasoning: str = Field(default="", description="Explanation of the decomposition")
    estimated_steps: int = Field(default=10, description="Estimated steps to complete")


# Default decomposition templates for vision-only mode
DEFAULT_DECOMPOSITIONS = {
    "find": TaskDecomposition(
        task="find an object",
        subgoals=[
            Subgoal(
                id="subgoal_1",
                action="navigate",
                target="target_object",
                description="Navigate to find the target object",
                dependencies=[]
            )
        ],
        reasoning="Simple search task: navigate until object is found",
        estimated_steps=10
    ),
    "pick up": TaskDecomposition(
        task="pick up an object",
        subgoals=[
            Subgoal(
                id="subgoal_1",
                action="navigate",
                target="target_object",
                description="Navigate to the target object",
                dependencies=[]
            ),
            Subgoal(
                id="subgoal_2",
                action="pickup",
                target="target_object",
                description="Pick up the target object",
                dependencies=["subgoal_1"]
            )
        ],
        reasoning="Pickup task: first navigate to object, then pick it up",
        estimated_steps=15
    ),
    "open": TaskDecomposition(
        task="open an object",
        subgoals=[
            Subgoal(
                id="subgoal_1",
                action="navigate",
                target="target_object",
                description="Navigate to the target object",
                dependencies=[]
            ),
            Subgoal(
                id="subgoal_2",
                action="open",
                target="target_object",
                description="Open the target object",
                dependencies=["subgoal_1"]
            )
        ],
        reasoning="Open task: navigate to object, then open it",
        estimated_steps=12
    ),
    "put": TaskDecomposition(
        task="put an object somewhere",
        subgoals=[
            Subgoal(
                id="subgoal_1",
                action="navigate",
                target="target_object",
                description="Navigate to the target object",
                dependencies=[]
            ),
            Subgoal(
                id="subgoal_2",
                action="pickup",
                target="target_object",
                description="Pick up the target object",
                dependencies=["subgoal_1"]
            ),
            Subgoal(
                id="subgoal_3",
                action="navigate",
                target="receptacle",
                description="Navigate to the receptacle",
                dependencies=["subgoal_2"]
            ),
            Subgoal(
                id="subgoal_4",
                action="put",
                target="receptacle",
                description="Put the object in the receptacle",
                dependencies=["subgoal_3"]
            )
        ],
        reasoning="Put task: navigate to object, pick it up, navigate to receptacle, put it down",
        estimated_steps=20
    )
}


class TaskDecomposer:
    """
    Decomposes high-level tasks into executable subgoals.

    Uses LangChain with an LLM to perform structured decomposition.
    When LLM is not available, uses template-based decomposition.

    Example:
        >>> decomposer = TaskDecomposer(use_llm=True, api_key="...")
        >>> decomposition = decomposer.decompose("Pick up the red mug")
        >>> for subgoal in decomposition.subgoals:
        ...     print(f"{subgoal.action}: {subgoal.target}")
    """

    # Prompt template for LLM decomposition
    DECOMPOSITION_PROMPT = """You are an embodied AI agent in a home environment.

## Current Environment State
{environment_section}

## Task
{task}

## CRITICAL RULES
1. Check if target is already visible in the environment
2. If target is visible and close (< 2m), plan ONLY: navigate to it
3. Keep plans MINIMAL - use FEWEST subgoals possible
4. Do NOT add exploration steps unless target is not visible
5. For "find X" tasks: ONE navigate subgoal if visible, otherwise explore
6. For "pick up X" tasks: navigate + pickup (2 subgoals max)
7. For "open X" tasks: navigate + open (2 subgoals max)

Available actions:
- navigate: Move to find/reach a target object or location
- pickup: Pick up an object (must be close to it first)
- put: Put down an object at a location
- open: Open an object (door, fridge, drawer, etc.)
- close: Close an object

Return your response as JSON with this structure:
{{
    "task": "original task",
    "subgoals": [
        {{
            "id": "subgoal_1",
            "action": "navigate",
            "target": "simple object name",
            "description": "brief description",
            "dependencies": []
        }}
    ],
    "reasoning": "why this minimal decomposition",
    "estimated_steps": 5
}}
"""

    def __init__(
        self,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2",
        temperature: float = 0.1,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the TaskDecomposer.

        Args:
            use_llm: Whether to use LLM for decomposition.
            api_key: OpenAI API key. Uses settings or env var if None.
            model: LLM model name.
            temperature: LLM temperature for generation.
            settings: Settings instance. Uses default_settings if None.
        """
        self._use_llm = use_llm
        self._model = model
        self._temperature = temperature
        self._settings = settings or default_settings

        # Get API key from settings or parameter
        self._api_key = api_key or self._settings.llm.api_key

        # LLM client (initialized lazily)
        self._llm = None

        if self._use_llm and self._api_key:
            self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LangChain LLM client."""
        try:
            from langchain_openai import ChatOpenAI
            import os

            # Get API base from settings or environment
            api_base = self._settings.llm.api_base if self._settings else None
            api_base = api_base or os.environ.get("OPENAI_API_BASE")

            llm_kwargs = {
                "model": self._model,
                "temperature": self._temperature,
                "api_key": self._api_key,
                "max_tokens": 1024
            }

            if api_base:
                llm_kwargs["base_url"] = api_base

            self._llm = ChatOpenAI(**llm_kwargs)
            logger.info(f"Initialized LLM for task decomposition: {self._model}")
        except ImportError:
            logger.warning(
                "langchain_openai not installed. Falling back to template-based decomposition."
            )
            self._llm = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}. Using template-based decomposition.")
            self._llm = None

    def decompose(
        self,
        instruction: str,
        environment: Optional[EnvironmentObservation] = None
    ) -> TaskDecomposition:
        """
        Decompose a task instruction into subgoals.

        Args:
            instruction: The task instruction to decompose.
            environment: Current environment observation (optional).

        Returns:
            TaskDecomposition with subgoals and metadata.
        """
        # Try LLM decomposition if available
        if self._use_llm and self._llm is not None:
            return self._decompose_with_llm(instruction, environment)

        # Fall back to template-based decomposition
        return self._decompose_with_templates(instruction, environment)

    def _decompose_with_llm(
        self,
        instruction: str,
        environment: Optional[EnvironmentObservation] = None
    ) -> TaskDecomposition:
        """
        Use LLM to decompose the task.

        Args:
            instruction: The task instruction.
            environment: Current environment observation.

        Returns:
            TaskDecomposition from LLM response.
        """
        try:
            # Format environment section
            if environment:
                env_section = environment.to_prompt_string()
            else:
                env_section = "No environment information available."

            prompt = self.DECOMPOSITION_PROMPT.format(
                task=instruction,
                environment_section=env_section
            )
            response = self._llm.invoke(prompt)

            # Parse JSON response
            content = response.content
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Convert to TaskDecomposition
            subgoals = [
                Subgoal(**sg) for sg in data.get("subgoals", [])
            ]

            decomposition = TaskDecomposition(
                task=data.get("task", instruction),
                subgoals=subgoals,
                reasoning=data.get("reasoning", ""),
                estimated_steps=data.get("estimated_steps", 10)
            )

            logger.info(f"LLM decomposition successful: {len(subgoals)} subgoals")
            return decomposition

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}. Using template-based.")
            return self._decompose_with_templates(instruction, environment)
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}. Using template-based.")
            return self._decompose_with_templates(instruction, environment)

    def _decompose_with_templates(
        self,
        instruction: str,
        environment: Optional[EnvironmentObservation] = None
    ) -> TaskDecomposition:
        """
        Use templates for decomposition (vision-only mode).

        Args:
            instruction: The task instruction.
            environment: Current environment observation.

        Returns:
            TaskDecomposition based on keyword matching.
        """
        instruction_lower = instruction.lower()

        # Extract target from instruction
        target = self._extract_target(instruction, "target_object")

        # Check if target is already visible and close
        if environment:
            visible_target = environment.has_visible_object(target, max_distance=2.0)
            if visible_target:
                obj_name = visible_target.get("name", visible_target.get("objectType", target))
                distance = visible_target.get("distance", 0)
                logger.info(f"Target '{target}' visible at {distance:.1f}m - simple plan")

                # Check if it's a pickup task
                if "pick" in instruction_lower or "拿" in instruction_lower or "捡" in instruction_lower:
                    return TaskDecomposition(
                        task=instruction,
                        subgoals=[
                            Subgoal(
                                id="subgoal_1",
                                action="navigate",
                                target=obj_name,
                                description=f"Navigate to {obj_name} ({distance:.1f}m away)",
                                dependencies=[]
                            ),
                            Subgoal(
                                id="subgoal_2",
                                action="pickup",
                                target=obj_name,
                                description=f"Pick up {obj_name}",
                                dependencies=["subgoal_1"]
                            )
                        ],
                        reasoning=f"Target {obj_name} visible at {distance:.1f}m",
                        estimated_steps=3
                    )
                else:
                    # Simple navigate task
                    return TaskDecomposition(
                        task=instruction,
                        subgoals=[
                            Subgoal(
                                id="subgoal_1",
                                action="navigate",
                                target=obj_name,
                                description=f"Navigate to {obj_name} ({distance:.1f}m away)",
                                dependencies=[]
                            )
                        ],
                        reasoning=f"Target {obj_name} visible at {distance:.1f}m",
                        estimated_steps=2
                    )

        # Match instruction to template
        for keyword, template in DEFAULT_DECOMPOSITIONS.items():
            if keyword in instruction_lower:
                # Create a copy with the actual instruction
                subgoals = [
                    Subgoal(
                        id=sg.id,
                        action=sg.action,
                        target=self._extract_target(instruction, sg.target),
                        description=sg.description,
                        dependencies=list(sg.dependencies)
                    )
                    for sg in template.subgoals
                ]

                decomposition = TaskDecomposition(
                    task=instruction,
                    subgoals=subgoals,
                    reasoning=f"Template-based decomposition using '{keyword}' template",
                    estimated_steps=template.estimated_steps
                )

                logger.info(f"Template decomposition: {len(subgoals)} subgoals")
                return decomposition

        # Default fallback: single navigate subgoal
        default = TaskDecomposition(
            task=instruction,
            subgoals=[
                Subgoal(
                    id="subgoal_1",
                    action="navigate",
                    target=instruction,
                    description=f"Complete task: {instruction}",
                    dependencies=[]
                )
            ],
            reasoning="Default decomposition: navigate to complete task",
            estimated_steps=15
        )

        logger.info("Using default decomposition: single navigate subgoal")
        return default

    def _extract_target(self, instruction: str, placeholder: str) -> str:
        """
        Extract target object from instruction.

        Args:
            instruction: The task instruction.
            placeholder: The placeholder target (e.g., "target_object").

        Returns:
            Extracted target or the instruction itself.
        """
        # Simple extraction: use the instruction as target
        # In a more sophisticated implementation, we would parse the instruction
        # to extract specific object names

        # For now, return a generic description
        if "target_object" in placeholder:
            return instruction
        return placeholder

    def is_available(self) -> bool:
        """
        Check if LLM-based decomposition is available.

        Returns:
            True if LLM is initialized, False otherwise.
        """
        return self._llm is not None
