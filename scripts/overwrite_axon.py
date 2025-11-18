# This script allows overwriting the axon IP and PORT for a specific miner.
# IMPORTANT: You must run the miner with a valid and public EXTERNAL_IP even you're going 
#            to overwrite it here.

import time

import bittensor as bt
import httpx


def gen_http_auth(hotkey: bt.Keypair) -> httpx.BasicAuth:
    timestamp = int(time.time())
    message = f"<Bytes>Eastworld AI {timestamp}</Bytes>"
    signature = hotkey.sign(data=message)

    return httpx.BasicAuth(
        username=f"{hotkey.ss58_address}|{timestamp}",
        password=signature.hex(),
    )


if __name__ == "__main__":
    # For mainnet use https://sn94.eastworld.ai/sn/axon
    endpoint = "https://tsn333.eastworld.ai/sn/axon"
    # The miner's wallet info
    wallet_name = "default"
    hotkey_name = "default"
    miner_uid = 123
    overwrite_ip = "1.2.3.4"  # The real ip of the miner. Empty string "" to reset overwrite
    overwrite_port = 12345    # The real port of the miner.

    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    body = {
        "uid": miner_uid,
        "ip": overwrite_ip,
        "port": overwrite_port,
    }

    print(f"Overwriting axon for miner: {miner_uid} {overwrite_ip}:{overwrite_port}")
    http_client = httpx.Client()
    req = http_client.build_request("POST", endpoint, json=body)
    res = http_client.send(req, auth=gen_http_auth(wallet.hotkey))
    if res.status_code != 204:
        print(f"Error: {res.status_code} - {res.text}")
    else:
        print("Axon overwritten successfully.")
