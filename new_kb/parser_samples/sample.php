<?php

namespace Groundwork\Samples;

require_once 'helpers.php';

class Animal
{
    protected string $name;
    private int $age;

    public function __construct(string $name, int $age)
    {
        $this->name = $name;
        $this->age = $age;
    }

    // base implementation
    public function speak(): string
    {
        return "{$this->name} makes a sound.";
    }
}

class Dog extends Animal
{
    public function speak(): string
    {
        return "{$this->name} barks.";
    }
}
