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

from pydantic import BaseModel


class EWItem(BaseModel):
    name: str
    description: str
    count: int


class EWObservation(BaseModel):
    lidar: list[tuple[str, ...]]
    odometry: tuple[str, ...]
    # Notable terrains features around.
    terrain: list[tuple[str, ...]]
    weather: list[tuple[str, ...]]
    # Agent's current location description.
    location: list[tuple[str, ...]]
    # Structures around.
    structure: list[tuple[str, ...]]
    static: list[tuple[str, ...]]
    dynamic: list[tuple[str, ...]]


class EWContext(BaseModel):
    # Agent integrity, energy level, etc. Will be implemented in the future.
    stats: dict
    # Items in Agent's inventory.
    item: list[EWItem]
    # Environment observation.
    observation: EWObservation
    # Environment interaction to Agent. Conversation started by others, environmental damage, etc.
    interaction: list[tuple[str, ...]]
    # Available function to call to perform actions.
    action: list[dict]
    # Action execution log of last round.
    log: list[str]
    # Agent's weighted score.
    reward: float = 0.0


class EWApiResponse(BaseModel):
    code: int
    message: str
    turns: int
    uid: int
    key: str
    context: EWContext | None = None
