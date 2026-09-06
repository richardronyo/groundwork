package com.groundwork.samples

import kotlin.math.sqrt

open class Shape(val area: Double) {
    open fun describe(): String = "Shape with area $area"
}

class Circle(private val radius: Double) : Shape(Math.PI * radius * radius) {
    override fun describe(): String = "Circle with radius $radius"
}

// Euclidean distance between two points
fun distance(x1: Double, y1: Double, x2: Double, y2: Double): Double {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
}
