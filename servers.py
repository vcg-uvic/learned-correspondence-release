# servers.py ---
#
# Filename: servers.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Apr  2 19:23:30 2018 (-0700)
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

import os


def is_computecanada():

    hostname = os.getenv("HOSTNAME")

    check = False
    check += hostname.startswith("cedar")
    check += hostname.startswith("gra")
    check += hostname.startswith("cdr")

    return check


def is_vcg_uvic():

    hostname = os.getenv("HOSTNAME")

    check = False
    check += hostname.startswith("kingwood")

    return check


def is_cvlab_epfl():

    hostname = os.getenv("HOSTNAME")

    check = False
    check += hostname.startswith("icc")

    return check
#
# servers.py ends here
