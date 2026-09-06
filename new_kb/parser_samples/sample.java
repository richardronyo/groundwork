package com.groundwork.samples;

import java.util.List;
import java.util.ArrayList;

public class Animal {
    protected String name;
    protected int age;

    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String speak() {
        return name + " makes a sound.";
    }
}

class Dog extends Animal {
    public Dog(String name, int age) {
        super(name, age);
    }

    @Override
    public String speak() {
        return name + " barks.";
    }
}
