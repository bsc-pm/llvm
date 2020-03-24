int main() {
    int x;
    #pragma oss task reduction(::operator+: x)
    {}
    #pragma oss task reduction(operator+: x)
    {}
    #pragma oss task reduction(+: x)
    {}
}

// CHECK: #pragma oss task reduction(::operator+: x)
// CHECK: #pragma oss task reduction(+: x)
// CHECK: #pragma oss task reduction(+: x)

