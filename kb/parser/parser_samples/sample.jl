module Sample

struct Point
    x::Int
    y::Int
end

# adds two points
function add(p1::Point, p2::Point)::Point
    return Point(p1.x + p2.x, p1.y + p2.y)
end

p = add(Point(1, 2), Point(3, 4))
println(p)

end # module
