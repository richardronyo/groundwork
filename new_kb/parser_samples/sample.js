import { readFile } from 'fs/promises';

/**
 * Adds two numbers together.
 * @param {number} a - first addend
 * @param {number} b - second addend
 * @returns {number} the sum
 */
function add(a, b) {
    return a + b;
}

class Counter {
    constructor(start = 0) {
        this.count = start;
    }

    increment() {
        this.count += 1;
        return this.count;
    }
}

export { add, Counter };
