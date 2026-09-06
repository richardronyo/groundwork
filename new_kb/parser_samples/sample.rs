use std::fmt;

pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }

    // adds two points
    pub fn add(&self, other: &Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let p1 = Point::new(1, 2);
    let p2 = Point::new(3, 4);
    println!("{}", p1.add(&p2));
}
