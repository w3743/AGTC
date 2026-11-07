"""Offline reference for zero-before-grant handover. No CAN or hardware I/O."""

from dataclasses import dataclass
from itertools import permutations


@dataclass
class Receiver:
    session: int
    generation: int = 0
    power: int = 0
    armed: bool = True

    def revoke(self, session, generation):
        if not self.armed or session != self.session or generation < self.generation:
            return False
        if generation > self.generation:
            self.generation = generation
            self.power = 0
        # A delayed duplicate must never ACK a zero that no longer exists.
        return self.power == 0

    def grant(self, session, generation, power):
        if self.armed and session == self.session and generation == self.generation:
            self.power = power

    def hard_stop(self):
        self.power = 0
        self.armed = False


class Coordinator:
    def __init__(self, session=1):
        self.session = session
        self.generation = 1
        self.zero_confirmed = set()

    def ack(self, side, session, generation):
        if side in (0, 1) and (session, generation) == (self.session, self.generation):
            self.zero_confirmed.add(side)

    def grant_ready(self):
        return self.zero_confirmed == {0, 1}


def replay(events):
    coordinator = Coordinator()
    receivers = [Receiver(1, power=400), Receiver(1)]
    ack_queue = []
    for event in events:
        side = int(event[-1])
        rx = receivers[side]
        action = event[:-1]
        if action == "zero" and rx.revoke(1, 1):
            ack_queue.append(side)
        elif action == "ack" and side in ack_queue:
            coordinator.ack(side, 1, 1)
        elif action == "old":
            rx.grant(1, 0, (400, 0)[side])
        elif action == "new" and coordinator.grant_ready():
            rx.grant(1, 1, (0, 400)[side])
        elif action == "stop":
            rx.hard_stop()
        assert sum(board.power for board in receivers) <= 400
        assert all(board.armed or board.power == 0 for board in receivers)
    return receivers


def main():
    events = ("zero0", "zero1", "ack0", "ack1", "old0", "new1", "stop1")
    # Every ordered subset includes all combinations of omission and reorder.
    count = 0
    for size in range(len(events) + 1):
        for order in permutations(events, size):
            replay(order)
            count += 1

    happy = ("zero0", "zero1", "ack0", "ack1", "new1")
    assert [rx.power for rx in replay(happy)] == [0, 400]
    assert replay(happy + ("zero1", "ack1", "stop1", "new1"))[1].power == 0
    for prefix in (("zero0", "ack0"), ("zero1", "ack1"), ("zero0", "zero1")):
        assert replay(prefix + ("new1",))[1].power == 0
    coordinator = Coordinator()
    coordinator.ack(0, 0, 1)
    coordinator.ack(1, 1, 0)
    coordinator.ack(0, 1, 1)
    coordinator.ack(0, 1, 1)
    assert not coordinator.grant_ready()
    # Reboot is disarmed and accepts no traffic as a way to ARM. A new
    # external ARM creates a fresh session; packets from the prior one fail.
    rebooted = Receiver(0, armed=False)
    assert not rebooted.revoke(1, 1)
    rebooted.grant(1, 0, 400)
    assert rebooted.power == 0
    fresh = Receiver(2)
    fresh.grant(1, 0, 400)
    assert not fresh.revoke(1, 1)
    assert fresh.power == 0
    restarted = Coordinator(session=2)
    for side in (0, 1):
        restarted.ack(side, 1, 1)
    assert not restarted.grant_ready()
    for side in (0, 1):
        assert Receiver(2).revoke(2, 1)
        restarted.ack(side, 2, 1)
    assert restarted.grant_ready()
    # Next generation rejects earlier grants and earlier zero commands.
    left, right = replay(happy)
    assert right.revoke(1, 2)
    assert left.revoke(1, 2)
    right.grant(1, 1, 400)
    assert not left.revoke(1, 1)
    assert (left.power, right.power) == (0, 0)
    left.grant(1, 2, 400)
    assert (left.power, right.power) == (400, 0)
    assert not left.revoke(1, 2)
    print(f"PASS: {count} ordered subsets; normal commit, duplicate/stale ACK, reboot and hard stop")


if __name__ == "__main__":
    main()
