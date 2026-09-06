#include <iostream>
#include <vector>

class Shape {
public:
    Shape(double area) : area_(area) {}
    virtual double getArea() const { return area_; }
protected:
    double area_;
};

class Circle : public Shape {
public:
    Circle(double radius) : Shape(3.14159 * radius * radius), radius_(radius) {}
private:
    double radius_;
};

// sums a vector of ints
int sum(const std::vector<int>& values) {
    int total = 0;
    for (int v : values) total += v;
    return total;
}
