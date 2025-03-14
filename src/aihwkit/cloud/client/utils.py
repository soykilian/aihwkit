# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Utilities for the AIHW Composer API."""

from os import getenv, path
from typing import Dict
from configparser import ConfigParser

DEFAULT_URL = "https://api-aihw-composer.draco.res.ibm.com"


class ClientConfiguration:
    """Helper for retrieving the user configuration.

    Utility for retrieving the user configuration. The API token will be
    retrieved from, in order of preference:

    1. the ``AIHW_API_TOKEN`` environment variable.
    2. the ``aihwkit.conf`` configuration file.
    3. if the API token could not be found, it will return ``None``.

    The API URL will be retrieved from, in order of preference:

    1. the ``aihwkit.conf`` configuration file.
    2. the ``DEFAULT_URL`` module variable.

    The ``aihwkit.conf`` file is read from:

    * the current working directory.
    * ``XDG_CONFIG_HOME`` ``/aihwkit`` (by default, ``~/.config/aihwkit``).
    """

    def __init__(self) -> None:
        self.stored_config = self.parse_config()

    def parse_config(self) -> Dict:
        """Read the configuration from a config file.

        Returns:
            A dictionary with the contents of the ``aihwkit.conf``
            configuration file.
        """
        parser = ConfigParser()
        parser.read(
            [
                "aihwkit.conf",
                path.expanduser(
                    "{}/aihwkit/aihwkit.conf".format(getenv("XDG_CONFIG_HOME", "~/.config"))
                ),
            ]
        )

        # Check that the expected section is present.
        if "cloud" in parser:
            return dict(parser["cloud"])

        return {}

    @property
    def token(self) -> str:
        """Return the user token."""
        return getenv("AIHW_API_TOKEN", self.stored_config.get("api_token", None))

    @property
    def url(self) -> str:
        """Return the API URL."""
        return self.stored_config.get("api_url", DEFAULT_URL)
