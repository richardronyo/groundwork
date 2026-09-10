require 'json'

class Animal
  attr_accessor :name

  def initialize(name, age)
    @name = name
    @age = age
  end

  # base implementation
  def speak
    "#{@name} makes a sound."
  end
end

class Dog < Animal
  def speak
    "#{@name} barks."
  end
end

dog = Dog.new("Rex", 3)
puts dog.speak
