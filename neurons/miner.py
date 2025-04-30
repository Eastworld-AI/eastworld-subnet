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

import time
import random

from dotenv import load_dotenv
import bittensor as bt

from eastworld.base.miner import BaseMinerNeuron
from eastworld.protocol import Observation
#from eastworld.miner.junior import JuniorAgent

from eastworld.miner.senior import SeniorAgent


class WanderAgent(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(WanderAgent, self).__init__(config=config)

    async def forward(self, synapse: Observation) -> Observation:
        """
        Processes the incoming 'Observation' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (eastworld.protocol.Observation): The synapse object containing the agent environment data.

        Returns:
            eastworld.protocol.Observation: The synapse object with a random direction 'move_in_direction' call

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        direction = self.wander(synapse)
        distance = random.randint(5, 30)
        synapse.action = [
            {
                "name": "move_in_direction",
                "arguments": {"direction": direction, "distance": distance},
            }
        ]
        return synapse

    def wander(self, synapse: Observation) -> str:
        """Wander base on LiDAR data"""
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
        weights = [1] * len(directions)

        try:
            readings = {}
            for data in synapse.sensor.lidar:
                readings[data[0]] = float(data[1].split("m")[0]) - 5.0

            for i, d in enumerate(directions):
                weights[i] = readings.get(d, 50.0) / 50.0  # Normalize weights

            # Avoid moving backwards
            odometry_direction = synapse.sensor.odometry[0]
            i = directions.index(odometry_direction)
            weights[(i + 3) % len(weights)] /= 3
            weights[(i + 4) % len(weights)] = 1e-6
            weights[(i + 5) % len(weights)] /= 3
        except Exception as e:
            pass

        choice = random.choices(directions, weights=weights, k=1)[0]
        return choice


# This is the main function, which runs the miner.
if __name__ == "__main__":
    load_dotenv()

    with SeniorAgent(slam_data=None) as miner:
        while True:
            bt.logging.info(f"Miner is running... {time.time()}")
            time.sleep(30)
