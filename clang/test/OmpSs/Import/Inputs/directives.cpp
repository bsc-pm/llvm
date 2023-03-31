void f() {
    #pragma oss taskwait
    #pragma oss release
    #pragma oss task
    {}
    #pragma oss critical
    #pragma oss task for
    for (int i=0;i<1;++i);
    #pragma oss taskiter
    for (int i=0;i<1;++i);
    #pragma oss taskloop
    for (int i=0;i<1;++i);
    #pragma oss taskloop for
    for (int i=0;i<1;++i);
    int e = 0;
    #pragma oss atomic
    e++;
}