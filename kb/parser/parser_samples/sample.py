from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def add(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


if __name__ == "__main__":
    print(distance(Point(0, 0), Point(3, 4)))
