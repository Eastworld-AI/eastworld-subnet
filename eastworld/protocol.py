# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

import bittensor as bt
import pydantic


class Item(pydantic.BaseModel):
    # Item name
    name: str
    # Item description
    description: str
    # Item count
    count: int


class Sensor(pydantic.BaseModel):
    # LiDAR data to represent obstacles in different directions. Example: ("north", "5.0m", "intense")
    lidar: list[tuple[str, ...]]
    # Odometry data to represent the agent's movement. Example: ("north", "5.0m")
    odometry: tuple[str, ...]


class Perception(pydantic.BaseModel):
    # Environment Perception Layer
    environment: str
    # Object Detection and Classification Layer
    objects: str
    # Interaction Perception Layer
    interactions: list[tuple[str, ...]]


class Observation(bt.Synapse):
    """ """

    # Agent stats
    stats: dict

    # Items in agent's inventory
    items: list[Item]

    # Environment observations. MUST set default None, Synapse creates dummy instance in the headers.
    sensor: Sensor | None = None
    perception: Perception | None = None

    # Feedback of last action
    action_log: list[str]

    # Available actions (function calls)
    action_space: list[dict]

    # Miner's action response
    action: list[dict]

    # Reward (not implemented so far)
    reward: float
