package com.groundwork.samples

import scala.collection.mutable.ListBuffer

case class Point(x: Int, y: Int) {
  def add(other: Point): Point = Point(x + other.x, y + other.y)
}

class Accumulator {
  private val points = ListBuffer[Point]()

  def add(p: Point): Unit = points += p

  def total: Point = points.foldLeft(Point(0, 0))(_ add _)
}

object Sample extends App {
  val acc = new Accumulator()
  acc.add(Point(1, 2))
  println(acc.total)
}
