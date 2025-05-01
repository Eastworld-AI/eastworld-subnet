# The MIT License (MIT)
# Copyright 2025 Eastworld AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import math
import os
import random
import traceback
from typing import Annotated, TypedDict

import bittensor as bt
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from eastworld.base.miner import BaseMinerNeuron
from eastworld.miner.slam.grid import ANONYMOUS_NODE_PREFIX
from eastworld.miner.slam.isam import ISAM2
from eastworld.protocol import Observation

SENSOR_MAX_RANGE = 50.0


class JSONFileMemory:
    memory: dict

    def __init__(self, file_path: str):
        self.file_path = file_path

        self.memory = None
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.memory = json.load(f)
            except json.JSONDecodeError:
                bt.logging.error("Memory file corrupted, creating new memory")

        if self.memory is None:
            self.memory = {
                "goals": [],
                "plans": [],
                "reflections": [],
                "logs": [],
            }

    def save(self):
        """Save memory to file"""
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def push_reflection(self, reflection: str):
        self.memory["reflections"].append(reflection)
        if len(self.memory["reflections"]) > 20:
            self.memory["reflections"] = self.memory["reflections"][-10:]

    def push_log(self, action: str):
        log = {
            "action": action.strip(),
            "feedback": "",
            "repeat_times": 1,
        }
        self.memory["logs"].append(log)
        if len(self.memory["logs"]) > 100:
            self.memory["logs"] = self.memory["logs"][-60:]

    def update_log(self, feedback: str):
        if not self.memory["logs"]:
            # Miner may have restarted and the last action is lost
            return

        last_log = self.memory["logs"][-1]
        if last_log["feedback"]:
            # The last log already has feedback, unexpected behavior
            return
        last_log["feedback"] = feedback.strip()

        # Try to merge the last two logs if they are the same
        if len(self.memory["logs"]) < 2:
            return
        previous_log = self.memory["logs"][-2]
        if (
            previous_log["action"] == last_log["action"]
            and previous_log["feedback"] == last_log["feedback"]
        ):
            # Merge the two logs with the same action and feedback
            previous_log["repeat_times"] += 1
            self.memory["logs"].pop()


class AgentState(TypedDict):
    observation: Annotated[Observation, lambda a, b: b or a]
    navigation_locations: Annotated[list[str], lambda a, b: b or a]
    reflection: Annotated[str, lambda a, b: b or a]
    action: Annotated[dict, lambda a, b: b or a]
    errors: Annotated[set[str], lambda a, b: a.union(b)]


class SeniorAgent(BaseMinerNeuron):
    directions = [
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    ]

    uid: int
    step: int
    graph: CompiledStateGraph
    slam: ISAM2
    llm: openai.AsyncOpenAI
    memory: JSONFileMemory

    local_action_space: list[dict] = []

    def __init__(
        self, config=None, slam_data: str = None, memory_file_path: str = "memory.json"
    ):
        super(SeniorAgent, self).__init__(config=config)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.step = 0

        # Load environment variables from .env file
        load_dotenv()

        self.graph = self._build_graph()

        if slam_data is None:
            self.slam = ISAM2(load_data=False, data_dir="slam_data")
        else:
            self.slam = ISAM2(load_data=True, data_dir=slam_data)

        # Initialize OpenAI client
        self.llm = openai.AsyncOpenAI(timeout=10)
        
        # Initialize Google Generative AI client
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            genai.configure(api_key=google_api_key)
        else:
            bt.logging.warning("GOOGLE_API_KEY not found in environment variables. Gemini models may not work properly.")
        
        self.model_small = "gemini-2.0-flash-lite"
        self.model_medium = "gemini-2.0-flash"
        self.model_large = "gemini-2.0-flash"

        prompt_dir = "eastworld/miner/prompts"
        self.landmark_annotation_step = 0
        self.landmark_annotation_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_landmark_annotation.txt")
        )
        self.after_action_review_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_after_action_review.txt")
        )
        self.grounding_learning_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_grounding_learning.txt")
        )
        self.objective_reevaluation_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_objective_reevaluation.txt")
        )
        self.action_selection_prompt = PromptTemplate.from_file(
            os.path.join(prompt_dir, "senior_action_selection.txt")
        )

        self.memory = JSONFileMemory(memory_file_path)
        if not self.memory.memory["goals"]:
            self._init_memory()

        with open("eastworld/miner/local_actions.json", "r") as f:
            self.local_action_space = json.load(f)

        # Variables for `maze_run`
        self.maze_run_explore_direction = "north"
        self.maze_run_counter = 0

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("Localization & Mapping", self.localization_mapping)
        graph_builder.add_node("Perception", self.perception)
        graph_builder.add_node("Landmark Annotation", self.landmark_annotation)

        graph_builder.add_node("After-Action Review", self.after_action_review)
        graph_builder.add_node("Grounding & Learning", self.grounding_learning)
        graph_builder.add_node("Objective Reevaluation", self.objective_reevaluation)
        graph_builder.add_node("Action Selection", self.action_selection)

        graph_builder.add_node("Action Execution", self.action_execution)

        graph_builder.add_edge(START, "Localization & Mapping")
        graph_builder.add_edge("Localization & Mapping", "Perception")
        graph_builder.add_edge("Localization & Mapping", "Landmark Annotation")
        graph_builder.add_edge("Landmark Annotation", END)
        graph_builder.add_edge("Perception", "After-Action Review")
        graph_builder.add_edge("After-Action Review", "Grounding & Learning")
        graph_builder.add_edge("After-Action Review", "Objective Reevaluation")
        graph_builder.add_edge("Grounding & Learning", END)
        graph_builder.add_edge("Objective Reevaluation", "Action Selection")
        graph_builder.add_edge("Action Selection", "Action Execution")
        graph_builder.add_edge("Action Execution", END)

        return graph_builder.compile()

    def _init_memory(self):
        self.memory.memory["goals"] = [
            "Work alongside your team to accomplish critical objectives",
            "Venture deep into the uncharted canyon to scavenge vital components for your mothership's repairs",
        ]
        self.memory.memory["plans"] = [
            "Talk to your team to understand the current situation and the objectives",
            "Explore unknown areas to supplement data for navigation systems",
        ]

    async def forward(self, synapse: Observation) -> Observation:
        self.step += 1
        config = RunnableConfig(
            configurable={"thread_id": f"step_{self.uid}_{self.step}"}
        )
        state = await self.graph.ainvoke(
            AgentState(observation=synapse, errors=set()), config
        )
        if state["errors"]:
            bt.logging.error(
                f"Errors in LLM Graph: {len(state['errors'])}. Fallback to random walk"
            )
            direction, distance = self.maze_run(synapse)
            action = {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        else:
            action = state["action"]

        self.memory.save()

        bt.logging.info(f">> Agent Action: {action}")
        synapse.action = [action]
        return synapse

    def localization_mapping(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Localization & Mapping")
        try:
            synapse: Observation = state["observation"]
            lidar_data = {}
            odometry = 0
            odometry_direction = ""

            for data in synapse.sensor.lidar:
                if data[1][-1] == "+":
                    lidar_data[data[0]] = max(
                        float(data[1].split("m")[0]), SENSOR_MAX_RANGE + 1
                    )
                else:
                    lidar_data[data[0]] = float(data[1].split("m")[0])
            odometry_direction = synapse.sensor.odometry[0]
            odometry = float(synapse.sensor.odometry[1].split("m")[0])

            if odometry > 0:
                try:
                    bt.logging.debug(
                        f"SLAM: {lidar_data} {odometry} {odometry_direction}"
                    )
                    self.slam.run_iteration(lidar_data, odometry, odometry_direction)

                    x, y, theta = self.slam.get_current_pose()
                    bt.logging.debug(f"SLAM: Update Navigation Topology: {x} {y}")
                    self.slam.grid_map.update_nav_topo(
                        self.slam.pose_index, x, y, allow_isolated=True
                    )
                except Exception as e:
                    bt.logging.error(f"SLAM Error: {e}")
                    traceback.print_exc()

            nav_nodes_labeled_all = [
                f"{node_id} : {node_data[3]}"
                for node_id, node_data in self.slam.grid_map.nav_nodes.items()
                if not node_id.startswith(ANONYMOUS_NODE_PREFIX)
            ]
            state["navigation_locations"] = nav_nodes_labeled_all
        except Exception as e:
            bt.logging.error(f"Localization & Mapping Error: {e}")
            traceback.print_exc()
            # Error is not added to state for purpose. So the process flow can continue
        finally:
            return state

    async def perception(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Perception")
        # Extract entities from `Observation.Perception` (or vision image), etc.
        return state

    async def landmark_annotation(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Landmark Annotation")
        try:
            synapse: Observation = state["observation"]
            if (
                self.step - self.landmark_annotation_step < 5
                or synapse.sensor.odometry[1] == "0m"
            ):
                # No need to annotate landmarks on every step
                return state

            self.landmark_annotation_step = self.step

            x, y, theta = self.slam.get_current_pose()
            nav_nodes = self.slam.grid_map.get_nav_nodes(x, y, 40.0)
            nav_nodes_labeled = [
                node_id
                for node_id in nav_nodes
                if not node_id.startswith(ANONYMOUS_NODE_PREFIX)
            ]

            prompt_context = {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "anonymous_landmark_count": len(nav_nodes) - len(nav_nodes_labeled),
                "labeled_landmark_count": len(nav_nodes_labeled),
                "labeled_landmark_list": ", ".join(nav_nodes_labeled),
                "labeled_landmark_all": "\n".join(
                    [f"  - {k}" for k in state["navigation_locations"]]
                ),
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
                "environment": synapse.perception.environment,
                "objects": synapse.perception.objects,
            }
            prompt = self.landmark_annotation_prompt.format(**prompt_context)
            bt.logging.debug(f"Landmark Annotation Prompt: {prompt}")
            response = await self._call_llm(
                self.model_small,
                messages=[{"role": "user", "content": prompt}],
            )
            bt.logging.debug(f"Landmark Annotation Response: {response}")

            node_data = response.choices[0].message.content.splitlines()
            node_id = node_data[0].strip()
            node_desc = node_data[1].strip() if len(node_data) > 1 else ""
            if node_id != "NA":
                self.slam.grid_map.update_nav_topo(
                    self.slam.pose_index,
                    x,
                    y,
                    node_id=node_id,
                    node_desc=node_desc,
                    allow_isolated=True,
                )
        except Exception as e:
            bt.logging.error(f"Landmark Annotation Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Landmark Annotation Error: {e}")
        finally:
            return state

    async def after_action_review(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> After-Action Review")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            if synapse.action_log:
                self.memory.update_log(synapse.action_log[0])

            recent_reflections = ""
            for idx, r in enumerate(self.memory.memory["reflections"][-3:]):
                recent_reflections += f"  {idx + 1}. {r}\n"
            recent_action_log = ""
            for idx, l in enumerate(self.memory.memory["logs"][-10:]):
                repeat_str = (
                    f" (repeated {l['repeat_times']} times)"
                    if l["repeat_times"] > 1
                    else ""
                )
                recent_action_log += f"\n  - Log {idx + 1}\n    Action: {l['action']} {repeat_str}\n    Result: {l['feedback']}"

            action_space = ""
            for act in [*synapse.action_space, *self.local_action_space]:
                action_space += (
                    f"  - {act['function']['name']}: {act['function']['description']}\n"
                )

            prompt_context = {
                "goals": "\n".join(self.memory.memory["goals"]),
                "plans": "\n".join(self.memory.memory["plans"]),
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
                "odometry_reading": f"  - {', '.join(synapse.sensor.odometry)}",
                "perception": f"{synapse.perception.environment}\n{synapse.perception.objects}",
                "interaction": "\n".join(
                    [f"  - {', '.join(x)}" for x in synapse.perception.interactions]
                ),
                "items": "\n".join(
                    [
                        f"  - {item.name} x{item.count}: {item.description.strip()}"
                        for item in synapse.items
                    ]
                ),
                "navigation_locations": "\n".join(
                    [f"  - {x}" for x in state["navigation_locations"]]
                ),
                "action_space": action_space,
                "recent_reflections": recent_reflections,
                "recent_action_log": recent_action_log,
            }
            prompt = self.after_action_review_prompt.format(**prompt_context)
            bt.logging.debug(f"After Action Review Prompt: {prompt}")
            response = await self._call_llm(
                self.model_large,
                messages=[{"role": "user", "content": prompt}],
            )

            reflection = response.choices[0].message.content.strip()
            bt.logging.debug(f"After Action Review Response: {reflection}")
            state["reflection"] = reflection
            self.memory.push_reflection(reflection)
        except Exception as e:
            bt.logging.error(f"After Action Review Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"After Action Review Error: {e}")
        finally:
            return state

    async def grounding_learning(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Grounding & Learning")
        if state["errors"]:
            return state

        # TODO: Implement grounding and learning with LancerDB or Chroma
        return state

    async def objective_reevaluation(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Objective Reevaluation")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            prompt_context = {
                "goals": "\n".join(self.memory.memory["goals"]),
                "plans": "\n".join(self.memory.memory["plans"]),
                "reflection": state["reflection"],
            }
            prompt = self.objective_reevaluation_prompt.format(**prompt_context)
            bt.logging.debug(f"Objective Reevaluation Prompt: {prompt}")
            response = await self._call_llm(
                self.model_large,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content.strip().split("\n\n")
            bt.logging.debug(f"Objective Reevaluation Response: {content}")
            new_goals = content[0].split("\n")
            new_plans = content[1].split("\n")
            self.memory.memory["goals"] = [
                goal.strip()
                for goal in new_goals
                if goal.strip() and not goal.strip().startswith("#")
            ]
            self.memory.memory["plans"] = [
                plan.strip()
                for plan in new_plans
                if plan.strip() and not plan.strip().startswith("#")
            ]
        except Exception as e:
            bt.logging.error(f"Objective Reevaluation Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Objective Reevaluation Error: {e}")
        finally:
            return state

    async def action_selection(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Action Selection")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            prompt_context = {
                "goals": "\n".join([f"  - {x}" for x in self.memory.memory["goals"]]),
                "plans": "\n".join([f"  - {x}" for x in self.memory.memory["plans"]]),
                "reflection": state["reflection"],
                "sensor_readings": "\n".join(
                    [f"  - {', '.join(items)}" for items in synapse.sensor.lidar]
                ),
                "perception": f"{synapse.perception.environment}\n{synapse.perception.objects}",
                "interaction": "\n".join(
                    [f"  - {', '.join(x)}" for x in synapse.perception.interactions]
                ),
                "items": "\n".join(
                    [
                        f"  - {item.name} x{item.count}: {item.description.strip()}"
                        for item in synapse.items
                    ]
                ),
                "navigation_locations": "\n".join(
                    [f"  - {x}" for x in state["navigation_locations"]]
                ),
            }
            prompt = self.action_selection_prompt.format(**prompt_context)
            bt.logging.debug(f"Action Selection Prompt: {prompt}")
            response = await self._call_llm(
                self.model_large,
                messages=[{"role": "user", "content": prompt}],
                tools=[*synapse.action_space, *self.local_action_space],
                tool_choice="auto",
            )

            bt.logging.debug(f"Action Selection Response: {response}")

            action = response.choices[0].message.tool_calls[0].function
            if action:
                parsed_action = {
                    "name": action.name,
                    "arguments": json.loads(action.arguments),
                }
                state["action"] = parsed_action
                self.memory.push_log(
                    f"{parsed_action['name']}, "
                    + ", ".join(
                        [f"{k}: {v}" for k, v in parsed_action["arguments"].items()]
                    )
                )
        except Exception as e:
            bt.logging.error(f"Action Selection Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Selection Error: {e}")
        finally:
            return state

    def action_execution(self, state: AgentState) -> AgentState:
        bt.logging.debug(">> Action Execution")
        if state["errors"]:
            return state

        try:
            synapse: Observation = state["observation"]

            action = state["action"]
            if not action:
                bt.logging.error("No action to execute")
                direction, distance = self.maze_run(synapse)

                bt.logging.info(f"Direction: {direction}, Distance: {distance}")
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
                return state

            if action["name"] == "explore_wall_following":
                direction, distance = self.maze_run(synapse)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
            elif action["name"] == "navigate_to":
                target = action["arguments"].get("target")
                direction, distance = self.navigate_to(synapse, target)
                state["action"] = {
                    "name": "move_in_direction",
                    "arguments": {"direction": direction, "distance": distance},
                }
        except Exception as e:
            bt.logging.error(f"Action Execution Error: {e}")
            traceback.print_exc()
            state["errors"].add(f"Action Execution Error: {e}")
        finally:
            return state

    async def _call_llm(self, model, messages, temperature=0.7, max_tokens=1000, tools=None, tool_choice=None):
        """
        Unified method to call either OpenAI or Google Gemini models based on the model name.
        
        Args:
            model (str): Model name (either OpenAI or Gemini)
            messages (list): List of message dictionaries with role and content
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens to generate
            tools (list): Optional list of tools for function calling
            tool_choice (str): Tool choice strategy ("auto" or "none")
            
        Returns:
            Object with response content accessible via .choices[0].message.content
        """
        try:
            if model.startswith("gemini"):
                # Convert OpenAI message format to Gemini format
                gemini_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        # Prepend system message to user message since Gemini doesn't have system role
                        if len(gemini_messages) > 0 and gemini_messages[0]["role"] == "user":
                            gemini_messages[0]["content"] = f"{msg['content']}\n\n{gemini_messages[0]['content']}"
                        else:
                            # If this is the first message, add as user message
                            gemini_messages.append({"role": "user", "content": msg["content"]})
                    else:
                        gemini_messages.append(msg)
                        
                model_name = model
                if model_name == "gemini-2.0-flash-lite":
                    model_name = "gemini-1.5-flash"
                elif model_name == "gemini-2.0-flash":
                    model_name = "gemini-1.5-pro"
                
                # Convert OpenAI tools format to Gemini tools format if provided
                gemini_tools = None
                if tools:
                    gemini_tools = []
                    for tool in tools:
                        gemini_tools.append({
                            "function_declarations": [{
                                "name": tool["function"]["name"],
                                "description": tool["function"].get("description", ""),
                                "parameters": tool["function"].get("parameters", {})
                            }]
                        })
                
                # Create Gemini model
                gemini_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                    tools=gemini_tools
                )
                
                # Create a chat session
                chat = gemini_model.start_chat(history=[])
                
                # For simplicity, we'll just send the last user message with tools
                last_user_message = None
                for msg in reversed(gemini_messages):
                    if msg["role"] == "user":
                        last_user_message = msg["content"]
                        break
                
                if last_user_message:
                    response = await chat.send_message_async(last_user_message)
                    
                    # Handle tool calls if present
                    if hasattr(response, "candidates") and response.candidates and hasattr(response.candidates[0], "content") and hasattr(response.candidates[0].content, "parts"):
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, "function_call"):
                                # Create a response object similar to OpenAI's format with tool calls
                                class GeminiToolResponse:
                                    class Choice:
                                        class Message:
                                            class ToolCall:
                                                def __init__(self, function):
                                                    self.function = function
                                                    self.id = "gemini-tool-call-1"
                                                    self.type = "function"
                                                
                                            def __init__(self, content, tool_calls=None):
                                                self.content = content
                                                self.tool_calls = tool_calls or []
                                                
                                        def __init__(self, text, tool_calls=None):
                                            self.message = self.Message(text, tool_calls)
                                            
                                    def __init__(self, text, tool_calls=None):
                                        self.choices = [self.Choice(text, tool_calls)]
                                
                                tool_calls = [
                                    GeminiToolResponse.Choice.Message.ToolCall({
                                        "name": part.function_call.name,
                                        "arguments": part.function_call.args
                                    })
                                ]
                                
                                return GeminiToolResponse(response.text, tool_calls)
                
                # Create a response object similar to OpenAI's format
                class GeminiResponse:
                    class Choice:
                        class Message:
                            def __init__(self, content):
                                self.content = content
                                self.tool_calls = []
                                
                        def __init__(self, text):
                            self.message = self.Message(text)
                            
                    def __init__(self, text):
                        self.choices = [self.Choice(text)]
                
                return GeminiResponse(response.text)
            else:
                # Use OpenAI
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                if tools:
                    kwargs["tools"] = tools
                    
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
                    
                return await self.llm.chat.completions.create(**kwargs)
                
        except Exception as e:
            bt.logging.error(f"LLM API call error: {e}")
            raise e

    def random_walk(self, synapse: Observation) -> tuple[str, float]:
        weights = [1] * len(self.directions)

        # Update weights based on lidar data
        if synapse.sensor.lidar:
            readings = {}
            for data in synapse.sensor.lidar:
                readings[data[0]] = float(data[1].split("m")[0]) - 5.0

            for i, d in enumerate(self.directions):
                weights[i] = readings.get(d, 0) / 50.0

        # Avoid moving backwards
        if synapse.sensor.odometry[1] != "0m":
            i = self.directions.index(synapse.sensor.odometry[0])
            weights[(i + len(weights) // 2) % len(weights)] = 1e-6

        bt.logging.debug(f"Direction Weight: {[f'{f:.04f}' for f in weights]}")

        choice = random.choices(self.directions, weights=weights, k=1)[0]
        distance = random.randint(5, 30)

        return choice, distance

    def maze_run(self, synapse: Observation) -> tuple[str, float]:
        r = [1] * len(self.directions)
        for data in synapse.sensor.lidar:
            r[self.directions.index(data[0])] = float(data[1].split("m")[0])

        l = len(self.directions)
        cdi = self.directions.index(self.maze_run_explore_direction)
        ldi = (cdi - 1) % l
        rdi = (cdi + 1) % l
        bt.logging.debug(
            f"Current: {self.maze_run_explore_direction} {cdi}, Readings: {[f'{f:.01f}' for f in r]}"
        )
        wall_dist = 12 if sum((r[ldi], r[cdi], r[rdi])) / 3 > 20 else 6
        if r[rdi] < wall_dist:
            self.maze_run_counter = 0
            if r[cdi] < wall_dist:
                # Block by front and right: Turn left
                while r[ldi] < wall_dist:
                    ldi = (ldi - 1) % l
                choice = self.directions[ldi]
                distance = 5
            else:
                # Walls on right: Go straight
                choice = self.directions[cdi]
                distance = r[cdi] // random.randint(2, 4)
        else:
            self.maze_run_counter += 1
            # No walls on right side: Turn right
            choice = self.directions[rdi]
            distance = r[rdi] // random.randint(2, 4)

        if self.maze_run_counter > 10:
            # Break cycle
            return self.random_walk(synapse)

        self.maze_run_explore_direction = choice
        return choice, distance

    def navigate_to(self, synapse: Observation, target_node: str) -> tuple[str, float]:
        node = self.slam.grid_map.nav_nodes.get(target_node)
        if node is None:
            bt.logging.error(f"Navigation target node {target_node} not found")
            return self.random_walk(synapse)

        current_x, current_y, _ = self.slam.get_current_pose()
        path = self.slam.grid_map.pose_navigation(
            current_x, current_y, node[0], node[1]
        )
        if path:
            next_node = path[1]
            direction = self._relative_direction(
                current_x, current_y, next_node[0], next_node[1]
            )
            distance = self._relative_distance(
                current_x, current_y, next_node[0], next_node[1]
            )
            return direction, distance
        else:
            bt.logging.error(f"No path found to target node {target_node}")
            return self.random_walk(synapse)

    def _relative_direction(
        self, origin_x: float, origin_y: float, target_x: float, target_y: float
    ) -> str:
        # Calculate the relative direction from origin to target
        dx = target_x - origin_x
        dy = target_y - origin_y
        angle = (180 + (180 / 3.14) * math.atan2(dy, dx)) % 360
        angle = (angle + 22.5) % 360
        index = int(angle / 45)
        return self.directions[index]

    def _relative_distance(
        self, origin_x: float, origin_y: float, target_x: float, target_y: float
    ) -> float:
        # Calculate the relative distance from origin to target
        return math.sqrt((target_x - origin_x) ** 2 + (target_y - origin_y) ** 2)
