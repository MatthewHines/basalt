"""The Engine Control Module: policy in front of one primary and at most one guest.

The contract this package exists to keep machine-checkable (ECM-W2-DESIGN.md, Matt's
words): robust, tough, easy to use — a reliable ping, documented, stable, handback assured.
Nothing in here talks to the engine except through client.py, and nothing changes state
without a decision-log entry.
"""
