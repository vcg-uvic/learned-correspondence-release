#!/usr/bin/env python3
# main.py ---
#
# Filename: main.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Apr  2 17:47:08 2018 (-0700)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:


# Load data

from __future__ import print_function

from config import get_config, print_usage
from data import load_data
from network import MyNetwork

eps = 1e-10
use3d = False
config = None

config, unparsed = get_config()

print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")


def main(config):
    """The main function."""

    # Initialize network
    mynet = MyNetwork(config)

    # Run propper mode
    if config.run_mode == "train":

        # Load data train and validation data
        data = {}
        data["train"] = load_data(config, "train")
        data["valid"] = load_data(config, "valid")

        # Run train
        mynet.train(data)

    elif config.run_mode == "test":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        #data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")

        # Run train
        mynet.test(data)

    elif config.run_mode == "test_simple":
        
        data = {}
        data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")
        
        mynet.test_simple(data)        

    elif config.run_mode == "comp":

        # This mode is for running comparison experiments. While cleaning the
        # code, I took many parts out to make the code cleaner, which may have
        # caused some issues. We are releasing this part just to help
        # researchers, but we will not provide any support for this
        # part. Please use at your own risk.
        data = {}
        data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")

        # Run train
        mynet.comp(data)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

#
# main.py ends here
