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


import asyncio
import time
import traceback
import random
from urllib.parse import urlparse

import bittensor as bt
import httpx
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from eastworld.base.validator import BaseValidatorNeuron
from eastworld.protocol import Observation, Sensor, Perception, Item
from eastworld.validator.models import EWApiResponse, EWContext, EWObservation
from eastworld.utils.uids import check_uid_availability


class Validator(BaseValidatorNeuron):
    """
    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    http_client: httpx.AsyncClient
    inactive_miners: dict

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.http_client = httpx.AsyncClient()
        self.inactive_miners = {}

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)
        await self.fetch_and_update_scores()

    async def forward(self):
        """
        The forward function is called by the validator every time step.

        It is responsible for querying the network and scoring the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

        """
        try:
            # Call Eastworld API to get miner's state and observation.
            res = await self.get_observation()

            if not res:
                bt.logging.error("No response from Eastworld API.")
                await asyncio.sleep(30)
                return

            if res.code == 429:
                bt.logging.info("The next turn is not available yet. Wait for 15s.")
                await asyncio.sleep(15)
                return
            elif res.code != 200:
                bt.logging.error(
                    f"Failed to get observation from Eastworld. {res.code} {res.message}"
                )
                await asyncio.sleep(30)
                return

            # Validate the UID and hotkey from the API.
            uid_is_available = check_uid_availability(
                self.metagraph, res.uid, self.config.neuron.vpermit_tao_limit
            )
            bt.logging.debug(
                f"UID {res.uid} {self.metagraph.axons[res.uid].hotkey} {uid_is_available}"
            )
            if not uid_is_available:
                bt.logging.info(f"UID {res.uid} from API is not available for mining.")
                await asyncio.sleep(1)
                return
            if res.key != self.metagraph.axons[res.uid].hotkey:
                bt.logging.info(
                    f"UID {res.uid} hotkey mismatch API:{res.key} Metagraph:{self.metagraph.axons[res.uid].hotkey}"
                )
                await asyncio.sleep(5)
                return

            # Skip the miner for a certain period if it is inactive.
            if self.config.subtensor.network in ("test", "local"):
                notuntil, interval = self.inactive_miners.get(res.uid, (0, 0))
                if notuntil and notuntil > time.time():
                    bt.logging.info(f"Skip for inactive miner #{res.uid}.")
                    return

            if not res.context:
                bt.logging.error(f"No context from Eastworld API.")
                return

            axon = self.metagraph.axons[res.uid]
            bt.logging.info(f"Selected miner UID {res.uid} AXON {axon.ip}:{axon.port}")
            miner_uids = np.array([res.uid])
            synapse = await self.create_synapse(res.context)

            # The dendrite client queries the network.
            timeout = self.config.neuron.timeout
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                deserialize=False,
                timeout=timeout,
            )

            # Log the results for monitoring purposes.
            bt.logging.trace(f"Received responses: {responses}")

            synapse: Observation = responses[0]
            # Add skip time for inactive miners.
            if self.config.subtensor.network in ("test", "local"):
                if synapse.is_failure or not len(synapse.action):
                    notuntil, interval = self.inactive_miners.get(
                        res.uid, (time.time(), 60)
                    )
                    # Increase the skip time by 3 minutes each time, up to 30 minutes.
                    interval = min(interval + 180, 1800)
                    self.inactive_miners[res.uid] = (notuntil + interval, interval)
                    bt.logging.info(
                        f"Inactive miner #{res.uid}. Skip for {interval} seconds."
                    )
                else:
                    self.inactive_miners.pop(res.uid, None)

            if synapse.is_failure or not len(synapse.action):
                bt.logging.warning(
                    f"Failed to get action from miner #{res.uid}. Code: {synapse.dendrite.status_code}"
                )
                return

            await self.submit_action(res.turns, res.uid, synapse)
        except httpx.ConnectError as e:
            # Eastworld server is down. Retry after 60 seconds.
            bt.logging.error(
                f"Failed to connect to Eastworld server: {e}. Retry after 60s."
            )
            await asyncio.sleep(60)
        except Exception as e:
            traceback.print_exc()
            await asyncio.sleep(10)
            raise e

    def gen_http_auth(self) -> httpx.BasicAuth:
        """Generates the HTTP Basic Auth object for the Eastworld API with validator hotkey."""
        keypair = self.wallet.hotkey
        timestamp = int(time.time())
        message = f"<Bytes>Eastworld AI {timestamp}</Bytes>"
        signature = keypair.sign(data=message)

        return httpx.BasicAuth(
            username=f"{keypair.ss58_address}|{timestamp}",
            password=signature.hex(),
        )

    async def get_observation(self) -> EWApiResponse:
        """Fetches the observation data from the Eastworld API."""
        endpoint_url = urlparse(self.config.eastworld.endpoint_url)
        endpoint = f"{endpoint_url.scheme}://{endpoint_url.netloc}/sn/env"
        req = self.http_client.build_request("GET", endpoint)
        r = await self.http_client.send(req, auth=self.gen_http_auth())
        if r.status_code != 200:
            raise Exception(
                f"Failed to get observation from Eastworld. {r.status_code}"
            )

        ob_data = r.json()
        try:
            response = EWApiResponse.model_validate(ob_data)
            return response
        except Exception as e:
            bt.logging.error(f"Failed to parse API response: {e}")
            raise

    async def submit_action(self, turns: int, uid: int, synapse: Observation):
        """ """
        endpoint_url = urlparse(self.config.eastworld.endpoint_url)
        endpoint = f"{endpoint_url.scheme}://{endpoint_url.netloc}/sn/step"

        for act in synapse.action:
            if not isinstance(act, dict):
                bt.logging.warning("Synapse Observation.action item is not a dict")
                return

        data = {
            "turns": turns,
            "uid": uid,
            "key": synapse.axon.hotkey,
            "action": synapse.action,
        }
        req = self.http_client.build_request("POST", endpoint, json=data)

        r = await self.http_client.send(req, auth=self.gen_http_auth())
        if r.status_code > 499:
            raise Exception(
                f"Failed to submit action ({uid}) to Eastworld server. {r.status_code}"
            )
        ob = r.json()
        if ob.get("code") != 200 and ob.get("code") != 400:
            raise Exception(
                f"Failed to submit action ({uid}) to Eastworld server. {ob.get('code')} {ob.get('message')}"
            )
        bt.logging.info(
            f"Action of miner UID {uid} in turn {turns} submitted successfully."
        )

    async def create_synapse(self, context: EWContext) -> Observation:
        ob = context.observation
        sensor = Sensor(lidar=ob.lidar, odometry=ob.odometry)
        perception = Perception(
            environment="", objects="", interactions=context.interaction
        )

        # Summarize perception with LLM
        perception_prompt = ""
        with open("eastworld/validator/prompts/perception.txt", "r") as f:
            prompt_tpl = f.read()
            perception_context = {
                "terrain": "\n".join([f"    - {', '.join(x)}" for x in ob.terrain])
                or "    N/A",
                "weather": "\n".join([f"    - {', '.join(x)}" for x in ob.weather])
                or "    N/A",
                "location": "\n".join([f"    - {', '.join(x)}" for x in ob.location])
                or "    N/A",
                "structure": "\n".join(
                    [f"    - {', '.join(x[:-1])}\n{x[-1]}" for x in ob.structure]
                )
                or "    N/A",
                "static_object": "\n".join(
                    [f"    - {', '.join(x[:-1])}\n{x[-1]}" for x in ob.static]
                )
                or "    N/A",
                "dynamic_object": "\n".join(
                    [f"    - {', '.join(x[:-1])}\n{x[-1]}" for x in ob.dynamic]
                )
                or "    N/A",
            }
            perception_prompt = prompt_tpl.format(**perception_context)

        async def get_llm_response(prompt):
            if not prompt:
                return ""
            async with httpx.AsyncClient() as client:
                llm = AsyncOpenAI(http_client=client, timeout=10)
                llm_model = self.config.eastworld.llm_model
                llm_args = {
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if llm_model.startswith("gpt-5") or llm_model.startswith("gemini-2.5"):
                    llm_args["reasoning_effort"] = "low"
                response = await llm.chat.completions.create(**llm_args)
                content = ""
                if response.choices[0].message.content:
                    content = response.choices[0].message.content.strip()
                bt.logging.trace(f"LLM response in perception: \n{content}")
                return content

        perception_content = await get_llm_response(perception_prompt)
        environment_content, objects_content = self._parse_perception_content(
            perception_content
        )

        # Update perception with LLM response
        perception.environment = environment_content
        perception.objects = objects_content

        items = [
            Item(name=x.name, description=x.description, count=x.count)
            for x in context.item
        ]

        synapse = Observation(
            stats=context.stats,
            items=items,
            sensor=sensor,
            perception=perception,
            action_log=context.log,
            action_space=context.action,
            action=[],
            reward=context.reward,
        )
        return synapse

    def _parse_perception_content(self, content: str) -> tuple[str, str]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        header_indices = []
        for i, line in enumerate(lines):
            if line.startswith("#"):
                header_indices.append(i)

        environment_lines = []
        objects_lines = []
        if not header_indices and lines:
            # If no headers are found, assume the first line is environment and the rest are objects.
            environment_lines = lines[:1]
            objects_lines = lines[1:]
        elif len(header_indices) == 1:
            # If only one header is found, assume it's the start of the objects section.
            environment_lines = lines[: header_indices[0]]
            objects_lines = lines[header_indices[0] + 1 :]
        else:
            # If two or more headers are found, use the first two to split the content.
            environment_lines = lines[: header_indices[1]]
            objects_lines = lines[header_indices[1] + 1 :]

        environment_content = "\n".join(
            [line for line in environment_lines if not line.startswith("#")]
        )
        objects_content = "\n".join(
            [line for line in objects_lines if not line.startswith("#")]
        )

        return environment_content, objects_content

    async def fetch_and_update_scores(self):
        """Fetch latest miners' scores from Eastworld server"""
        endpoint_url = urlparse(self.config.eastworld.endpoint_url)
        endpoint = f"{endpoint_url.scheme}://{endpoint_url.netloc}/sn/score"

        req = self.http_client.build_request("GET", endpoint)
        r = await self.http_client.send(req, auth=self.gen_http_auth())
        if r.status_code != 200:
            raise Exception(
                f"Failed to get miner scores from Eastworld. {r.status_code} {r.text}"
            )

        data = r.json()
        uids = [d["uid"] for d in data["scores"]]
        new_scores = [d["score"] for d in data["scores"]]

        # Check if rewards contains NaN values.
        if np.isnan(new_scores).any():
            bt.logging.warning(f"NaN values detected in rewards: {new_scores}")
            # Replace any NaN values in scores with 0.
            new_scores = np.nan_to_num(new_scores, nan=0)

        # Ensure new_scores is a numpy array.
        new_scores = np.asarray(new_scores)
        uids_array = np.array(uids)

        # Handle edge case: If either new_scores or uids_array is empty.
        if new_scores.size == 0 or uids_array.size == 0:
            bt.logging.info(f"new_scores: {new_scores}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either new_scores or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of new_scores and uids_array match.
        if new_scores.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: new_scores array of shape {new_scores.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # If things work as expected, scores from Eastworld API should be same size as metagraph.
        # Exception case:
        #   The server synchronizes the network before the validator and increases the UIDs (metagraph.n). The
        #   code will raise an IndexError error. It won't happen if UID number reaches 256.
        # shape: [ metagraph.n ]
        scattered_scores: np.ndarray = np.zeros_like(self.scores)
        scattered_scores[uids_array] = new_scores

        # Update local scores.
        # shape: [ metagraph.n ]
        self.scores: np.ndarray = scattered_scores
        bt.logging.debug(f"Updated scores: \n{self.scores}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    load_dotenv()

    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator is running... {time.time()}")
            time.sleep(30)
