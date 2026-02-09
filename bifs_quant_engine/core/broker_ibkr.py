"""Placeholder for Interactive Brokers (IBKR) broker integration."""

from __future__ import annotations


class IBKRBroker:
    def __init__(self, *_, **__):
        self.connected = False

    def connect(self) -> None:
        # Placeholder: not actually connecting
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected
