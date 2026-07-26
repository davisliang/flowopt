"""Choosing between workflows on the two axes that matter: accuracy and cost.

Each function takes the items to compare and `on`, which says WHICH scores to
read — `on=DEV` to compare candidates on the dev split, `on=TEST` on test.
Naming the split at the call site is the point: an object that reports "its"
accuracy has to pick one silently, and a score that changes meaning partway
through a run is a good way to compare two different things by accident.
"""
from typing import Callable, Optional

# The two split accessors. Defined here — the one module both the optimizer and
# the designer already import — so neither has to define its own copy or import
# the other for it (designer ← optimizer would be a cycle).
DEV = lambda candidate: candidate.dev      # noqa: E731
TEST = lambda candidate: candidate.test    # noqa: E731


def _identity(item):
    """Default accessor: the item already carries `.accuracy` and `.cost`."""
    return item


def pareto_front(items: list, on: Callable = _identity) -> list:
    """Select the non-dominated items — the ones actually worth choosing between.

    An item is dominated if some other item is at least as accurate AND at least
    as cheap, and strictly better on at least one of the two.

    Args:
        items: The candidates or scores to compare.
        on: Maps an item to the object holding `.accuracy` and `.cost`. Use `DEV`
            or `TEST` above to pick a split.

    Returns:
        The non-dominated items, cheapest first, as the same objects passed in.
    """
    front = []
    for item in items:
        mine = on(item)
        dominated = any(
            other is not item
            and on(other).cost <= mine.cost
            and on(other).accuracy >= mine.accuracy
            and (on(other).cost < mine.cost or on(other).accuracy > mine.accuracy)
            for other in items)
        if not dominated:
            front.append(item)
    return sorted(front, key=lambda item: on(item).cost)


def best_under_budget(items: list, max_cost: float, on: Callable = _identity) -> Optional[object]:
    """Find the most accurate item you can afford.

    Args:
        items: The candidates or scores to compare.
        max_cost: Most you will pay per query, in USD.
        on: Maps an item to the object holding `.accuracy` and `.cost`.

    Returns:
        The most accurate item costing no more than `max_cost`, or None if
        everything costs more.
    """
    affordable = [item for item in items if on(item).cost <= max_cost]
    return max(affordable, key=lambda item: on(item).accuracy, default=None)


def cheapest_above_accuracy(items: list, min_accuracy: float,
                            on: Callable = _identity) -> Optional[object]:
    """Find the cheapest item that is still good enough.

    Args:
        items: The candidates or scores to compare.
        min_accuracy: The accuracy floor, in [0, 1].
        on: Maps an item to the object holding `.accuracy` and `.cost`.

    Returns:
        The cheapest item reaching `min_accuracy`, or None if none does.
    """
    good_enough = [item for item in items if on(item).accuracy >= min_accuracy]
    return min(good_enough, key=lambda item: on(item).cost, default=None)
