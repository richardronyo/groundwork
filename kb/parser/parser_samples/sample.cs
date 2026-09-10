using System;
using System.Collections.Generic;

namespace Groundwork.Samples
{
    public class Animal
    {
        public string Name { get; set; }
        protected int Age;

        public Animal(string name, int age)
        {
            Name = name;
            Age = age;
        }

        // virtual so Dog can override it
        public virtual string Speak() => $"{Name} makes a sound.";
    }

    public class Dog : Animal
    {
        public Dog(string name, int age) : base(name, age) { }

        public override string Speak() => $"{Name} barks.";
    }
}
