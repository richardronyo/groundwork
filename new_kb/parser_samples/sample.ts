import { readFileSync } from 'fs';

interface Shape {
    area(): number;
}

class Circle implements Shape {
    constructor(private radius: number) {}

    area(): number {
        return Math.PI * this.radius * this.radius;
    }
}

// sums the area of every shape
function totalArea(shapes: Shape[]): number {
    return shapes.reduce((sum, s) => sum + s.area(), 0);
}

export { Circle, totalArea };
