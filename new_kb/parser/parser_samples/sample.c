#include <stdio.h>

typedef struct {
    int x;
    int y;
} Point;

/* add two ints */
int add(int a, int b) {
    return a + b;
}

int main(void) {
    Point p = {1, 2};
    printf("%d\n", add(p.x, p.y));
    return 0;
}
