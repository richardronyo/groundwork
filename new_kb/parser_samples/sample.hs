module Sample where

-- A 2D point
data Point = Point { x :: Int, y :: Int } deriving (Show)

add :: Point -> Point -> Point
add p1 p2 = Point (x p1 + x p2) (y p1 + y p2)

main :: IO ()
main = print (add (Point 1 2) (Point 3 4))
