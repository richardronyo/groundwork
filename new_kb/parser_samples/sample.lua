local Point = {}
Point.__index = Point

function Point.new(x, y)
    local self = setmetatable({}, Point)
    self.x = x
    self.y = y
    return self
end

-- adds two points
function Point:add(other)
    return Point.new(self.x + other.x, self.y + other.y)
end

local p1 = Point.new(1, 2)
local p2 = Point.new(3, 4)
print(p1:add(p2).x)
