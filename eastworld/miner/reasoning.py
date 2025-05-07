# The MIT License (MIT)
# Copyright © 2025 Eastworld AI

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

import asyncio
import datetime
import threading
import time
import traceback
from collections import deque

import bittensor as bt
import httpx
import json_repair
from openai import OpenAI, AsyncOpenAI, APITimeoutError
from pydantic import BaseModel

from eastworld.protocol import Observation
from eastworld.base.miner import BaseMinerNeuron


action_standby = {
    "type": "function",
    "function": {
        "name": "standby",
        "description": "Stand by, waiting to reflect and decide your next move",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class ActionLog(BaseModel):
    timestamp: datetime.datetime
    action: str
    feedback: str
    repeat_times: int


class ReasoningAgent(BaseMinerNeuron):
    agent_uid: int

    memory_reflection: deque
    memory_action: deque

    prompt_system_tpl: str
    prompt_reflection_tpl: str
    prompt_action_tpl: str

    goals: list[str]
    plans: list[str]
    observations: deque
    reflection_step: int

    reflection_model: str
    action_model: str

    http_client: httpx.AsyncClient

    reasoning_thread: threading.Thread
    reasoning_should_exit: bool = False
    reasoning_should_reflect: bool = False

    def __init__(
        self,
        config=None,
        reflection_model="o4-mini",
        action_model="gpt-4o-mini",
        reflection_step=5,
    ):
        super(ReasoningAgent, self).__init__(config=config)

        self.agent_uid = self.metagraph.hotkeys.index(self.axon.info().hotkey)

        self.memory_reflection = deque(maxlen=10)
        self.memory_action = deque(maxlen=100)

        self.goals = []
        self.plans = []
        self.observations = deque(maxlen=20)
        self.reflection_step = reflection_step

        self.reflection_model = reflection_model
        self.action_model = action_model

        with open("eastworld/miner/prompts/reasoning_system.txt", "r") as f:
            self.prompt_system_tpl = f.read()
        with open("eastworld/miner/prompts/reasoning_reflection.txt", "r") as f:
            self.prompt_reflection_tpl = f.read()
        with open("eastworld/miner/prompts/reasoning_action.txt", "r") as f:
            self.prompt_action_tpl = f.read()

        # While hardcoded goals is not our preferred approach, consider it's a gift to the
        # new miners. Ensuring their initial baseline matches that of experienced players.
        self.goals = [
            "[medium] Compete with other agents by completing tasks to achieve superior performance",
            "[high] Explore the surroundings to discover Power Generators, Gears, Plants",
            "[medium] Collect Gear or Lumen Fungus then deliver to Quinn",
        ]

        self.http_client = httpx.AsyncClient()

    def __enter__(self):
        super().__enter__()

        self.reasoning_thread = threading.Thread(
            target=self._reflection_loop, daemon=True
        )
        self.reasoning_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reasoning_should_exit = True
        if hasattr(self, "reasoning_thread") and self.reasoning_thread.is_alive():
            self.reasoning_thread.join(5.0)

        return super().__exit__(exc_type, exc_value, traceback)

    def _reflection_loop(self):
        while not self.reasoning_should_exit:
            try:
                if (
                    not self.reasoning_should_reflect
                    and len(self.observations) < self.reflection_step
                ):
                    time.sleep(5)
                    continue

                self.reasoning_should_reflect = False
                self.reflection()
            except Exception as e:
                bt.logging.error(f"Reflection loop error: {e}")
                traceback.print_exc()
            except (asyncio.CancelledError, KeyboardInterrupt):
                break

    def push_reflection_memory(self, reflection: str):
        self.memory_reflection.append(reflection)

    def push_action_memory(self, action: str):
        action_log = ActionLog(
            timestamp=datetime.datetime.now(),
            action=action.strip(),
            feedback="",
            repeat_times=1,
        )
        self.memory_action.append(action_log)

    def update_action_memory(self, feedback: str):
        """
        This function updates the feedback of the last action in the memory.

        The action log is added to the memory immediately after the submission. But the action
        result is only available in the next observation.
        """
        if not self.memory_action:
            # Miner may have restarted and the last action is lost
            return

        last_log = self.memory_action[-1]
        if last_log.feedback:
            # The last log already has feedback, unexpected behavior
            return
        last_log.feedback = feedback.strip()

        # Try to merge the last two logs if they are the same
        if len(self.memory_action) < 2:
            return
        previous_log = self.memory_action[-2]
        if (
            previous_log.action == last_log.action
            and previous_log.feedback == last_log.feedback
        ):
            # Merge the two logs with the same action and feedback
            previous_log.timestamp = last_log.timestamp
            previous_log.repeat_times += 1
            self.memory_action.pop()

    async def forward(self, synapse: Observation) -> Observation:
        bt.logging.info(f"Feedback of previous action: {synapse.action_log}")
        self.update_action_memory("\n\n".join(synapse.action_log))
        self.observations.append(synapse)

        if not self.plans:
            bt.logging.warning("No plan available, using standby action.")
            self.reasoning_should_reflect = True
            synapse.action = [self.get_standby_action()]
            return synapse

        lidar = ""
        for items in synapse.sensor.lidar:
            lidar += f"  - {', '.join(items)}\n"
        odometry = f"  - {', '.join(synapse.sensor.odometry)}\n"
        perception = synapse.perception.environment + "\n" + synapse.perception.objects

        if len(synapse.items):
            items = "\n".join(
                [
                    f"  - {x.name}, Amount {x.count}, Description: {x.description.strip()}"
                    for x in synapse.items
                ]
            )
        else:
            items = "EMPTY"

        recent_action = ""
        for idx, l in enumerate(list(self.memory_action)[-5:]):
            l: ActionLog
            repeat_str = (
                f" (repeated {l.repeat_times} times)" if l.repeat_times > 1 else ""
            )
            recent_action += f"""
  - Log {idx + 1}
    Action: {l.action} {repeat_str}
    Result: {l.feedback}
"""

        llm_client = AsyncOpenAI(http_client=self.http_client)
        try:
            action_context = {
                "plan": self.plans[0],
                "lidar": lidar,
                "odometry": odometry,
                "perception": perception,
                "items": items,
                "recent_action": recent_action,
            }
            messages = [
                {
                    "role": "system",
                    "content": self.prompt_system_tpl.format(),
                },
                {
                    "role": "user",
                    "content": self.prompt_action_tpl.format(**action_context),
                },
            ]
            action_space = [action_standby, *synapse.action_space]
            bt.logging.debug(messages[1]["content"], ">>>> Action Prompt")
            # bt.logging.trace(action_space, ">>>> Action Space")
            response = await llm_client.chat.completions.create(
                model=self.action_model,
                messages=messages,
                tools=action_space,
                tool_choice="required",  # change to "auto" if your provider does not support "required"
                max_completion_tokens=1024,
                timeout=10,
            )

            if response.choices[0].finish_reason != "tool_calls":
                bt.logging.warning(f"LLM tool call failed: {response}")
                synapse.action = [self.get_standby_action()]
                return synapse

            action = response.choices[0].message.tool_calls[0].function
            bt.logging.trace(action, ">>>> Action: ")
            if action and action.name != "standby":
                parsed_action = {
                    "name": action.name,
                    "arguments": json_repair.loads(action.arguments),
                }
                synapse.action = [parsed_action]
                self.push_action_memory(
                    f"{action.name}, "
                    + ", ".join(
                        [f"{k}: {v}" for k, v in parsed_action["arguments"].items()]
                    )
                )
            else:
                self.reasoning_should_reflect = True
                synapse.action = [self.get_standby_action()]
                self.push_action_memory("standby, ")
        except APITimeoutError as e:
            bt.logging.error(f"API Timeout Error: {e}")
        except Exception as e:
            traceback.print_exc()
        finally:
            return synapse

    def get_standby_action(self):
        return {
            "name": "talk_to",
            "arguments": {
                "target": f"Agent {self.agent_uid}",
                "content": "Think, Lara... there has to be a way.",
            },
        }

    def reflection(self):
        obs = [self.observations.popleft() for _ in range(len(self.observations))]
        if not obs:
            bt.logging.warning("No observations to reflect on.")
            return

        goals = "\n".join([f" - {x}" for x in self.goals])
        plans = "\n".join([f" - {x}" for x in self.plans])

        lidar = "\n".join([f"  - {', '.join(items)}" for items in obs[-1].sensor.lidar])
        odometry = "\n".join([f"  - {', '.join(ob.sensor.odometry)}" for ob in obs])
        perception = obs[-1].perception.environment + "\n" + obs[-1].perception.objects

        if len(obs[-1].items):
            items = "\n".join(
                [
                    f"  - {x.name}, Amount {x.count}, Description: {x.description.strip()}"
                    for x in obs[-1].items
                ]
            )
        else:
            items = "EMPTY"

        action_space = ""
        for act in obs[-1].action_space:
            action_space += (
                f"  - {act['function']['name']}: {act['function']['description']}\n"
            )

        previous_reflection = (
            self.memory_reflection[-1] if self.memory_reflection else "N/A"
        )
        recent_action = ""
        for idx, l in enumerate(list(self.memory_action)[-len(obs) : -1]):
            l: ActionLog
            repeat_str = (
                f" (repeated {l.repeat_times} times)" if l.repeat_times > 1 else ""
            )
            recent_action += f"""
  - Log {idx + 1}
    Action: {l.action} {repeat_str}
    Result: {l.feedback}
"""

        llm_client = OpenAI()
        try:
            reflection_context = {
                "goals": goals,
                "plans": plans,
                "lidar": lidar,
                "odometry": odometry,
                "perception": perception,
                "items": items,
                "action_space": action_space,
                "previous_reflection": previous_reflection,
                "recent_action": recent_action,
            }
            messages = [
                {
                    "role": "user",
                    "content": self.prompt_reflection_tpl.format(**reflection_context),
                },
            ]
            bt.logging.debug(messages[0]["content"], ">>>> Reflection Prompt")

            t1 = time.time()
            # Notice we're using responses api here, instead of chat completions.
            # Reasoning models from different providers have various output formats and
            # API. You need to change the code to fit in your use case.
            response = llm_client.responses.create(
                model=self.reflection_model,
                reasoning={"effort": "medium"},
                input=messages,
                max_output_tokens=10240,
                timeout=60,
            )
            t2 = time.time()
            bt.logging.trace(f"Reflection time: {t2 - t1:.2f} seconds")
            # bt.logging.trace(response, ">>>> Reflection Response")

            if not response.status == "completed" or not response.output_text:
                bt.logging.warning(f"LLM generation failed: {response}")
                return

            output = json_repair.loads(response.output_text)
            reflection = output.get("reflection", "")
            goals = output.get("goals", [])
            plans = output.get("plans", [])

            bt.logging.debug(f"Updated Reflection: {reflection}")
            bt.logging.debug(f"Updated Goals: {goals}")
            bt.logging.debug(f"Updated Plans: {plans}")

            if reflection:
                self.push_reflection_memory(reflection)
            if goals:
                self.goals = goals
            if plans:
                self.plans = plans
        except APITimeoutError as e:
            bt.logging.error(f"API Timeout Error: {e}")
        except Exception as e:
            traceback.print_exc()
