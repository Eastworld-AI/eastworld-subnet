# Calculate slippage when staking TAO into a subnet
# Usage: python calculate_slippage.py --netuid <NETUID> --tao <AMOUNT> [--network <NETWORK>]

import argparse
import sys
import traceback
import bittensor as bt
from bittensor.utils.balance import Balance


def calculate_slippage(netuid: int, tao_amount: float, network: str = "finney"):
    """
    Calculates the slippage and price impact of staking TAO into a specific subnet.
    """
    print(f"Connecting to network: {network}...")

    try:
        # Connect to network
        subtensor = bt.Subtensor(network=network)

        # Check if subnet exists
        if not subtensor.subnet_exists(netuid):
            print(f"Error: Subnet with Netuid {netuid} does not exist.")
            return

        subnet_info = subtensor.subnet(netuid=netuid)

        if subnet_info is None:
            print(f"Error: Could not retrieve information for subnet {netuid}.")
            return

    except Exception as e:
        print(f"Failed to connect or retrieve subnet info: {e}")
        return

    # Input validation
    if tao_amount <= 0:
        print("Error: TAO amount must be positive.")
        return

    try:
        # Convert float to Balance object
        tao = Balance.from_tao(tao_amount)

        # Calculate ideal conversion (Zero Slippage)
        alpha_ideal = subnet_info.tao_to_alpha(tao)
        current_price = subnet_info.price

        print(f"\n--- Subnet {netuid} Staking Analysis ({tao}) ---")
        print(f"Current Price: {current_price}")
        print(f"Ideal Alpha (No Slippage): {alpha_ideal}")

        # Calculate slippage percentage
        slippage_percentage = subnet_info.tao_to_alpha_with_slippage(
            tao, percentage=True
        )
        print(f"Expected Slippage: {slippage_percentage:.4f}%")

        # Get detailed breakdown (Actual received vs Slippage loss)
        alpha_received, slippage_amount = subnet_info.tao_to_alpha_with_slippage(tao)

        print(f"Actual Alpha Received: {alpha_received}")
        print(f"Slippage Amount: {slippage_amount}")

        # Calculate price impact after staking
        # Note: Check for attributes to ensure compatibility with different SDK versions
        if hasattr(subnet_info, "tao_in") and hasattr(subnet_info, "alpha_in"):
            new_tao_in = subnet_info.tao_in.tao + tao_amount
            new_alpha_in = subnet_info.alpha_in.tao - alpha_received.tao

            if new_alpha_in <= 0:
                print(
                    "Warning: Calculated remaining Alpha pool is invalid. Skipping price prediction."
                )
            else:
                price_after = Balance.from_tao(new_tao_in / new_alpha_in)
                print(f"Estimated Price after Stake: {price_after}")

                if current_price.tao > 0:
                    rate_tolerance = (price_after.tao / current_price.tao) - 1
                    print(f"Minimum Rate Tolerance: {rate_tolerance:.6f}")
        else:
            print(
                "Info: Current SDK version does not expose pool data in 'subnet_info'. Cannot calculate price impact."
            )

    except Exception as e:
        print(f"An error occurred during calculation: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate slippage and price impact when staking TAO into a Bittensor subnet."
    )

    parser.add_argument(
        "--netuid",
        type=int,
        required=True,
        help="The Subnet ID (Netuid) to stake into.",
    )
    parser.add_argument(
        "--tao", type=float, required=True, help="The amount of TAO to stake."
    )
    parser.add_argument(
        "--network",
        type=str,
        default="finney",
        help="The Bittensor network to connect to (default: finney, options: test, local).",
    )

    args = parser.parse_args()

    calculate_slippage(args.netuid, args.tao, args.network)


if __name__ == "__main__":
    main()
