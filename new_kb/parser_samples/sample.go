package sample

import (
	"fmt"
)

type Point struct {
	X int
	Y int
}

// Add returns the sum of two points.
func (p Point) Add(other Point) Point {
	return Point{X: p.X + other.X, Y: p.Y + other.Y}
}

func main() {
	p1 := Point{X: 1, Y: 2}
	p2 := Point{X: 3, Y: 4}
	fmt.Println(p1.Add(p2))
}
