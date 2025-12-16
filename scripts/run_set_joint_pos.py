#!/usr/bin/env python3

import argparse
import sys
import os
import time

# Add the parent directory to the Python path so we can import orca_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orca_core import OrcaHand

def main():
    parser = argparse.ArgumentParser(description='Move OrcaHand to neutral position.')
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the orcahand model folder (e.g., /path/to/orcahand_v1)"
    )
    
    args = parser.parse_args()

    try:
        # Initialize the hand
        hand = OrcaHand(model_path=args.model_path)
            
        # Connect to the hand
        success, message = hand.connect()
        if not success:
            print(f"Failed to connect: {message}")
            return 1
            
        print("Connected to hand successfully")
        
        # Enable torque
        hand.enable_torque()
        print("Torque enabled")
        print("Available motor IDs:", hand.motor_ids)

        # Set joint position goal
        while True:
            try:
                # need to input list of different angles for each joint
                # set_join_pos reads a list of angles as in order of joint IDs available in config
                pos_goal = list(map(int, input("Enter target position (-1000 to exit): ").split()))
            except ValueError:
                print("Please enter three integers.")
                continue

            if -1000 in pos_goal:
                break

            # if target_position <  or target_position > :
            #     print("Position must be within joint ROMs.")
            #     continue

            # Move to joint position goal
            print("Moving to joint position goal...")
            hand.set_joint_pos(pos_goal, num_steps=50, step_size=0.005)
            print("Reached joint position goal")

            time.sleep(3)


        # Disable torque and disconnect
        hand.disable_torque()
        hand.disconnect()
        print("Disconnected from hand")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 