// RUN: %oss-compile-and-run
// RUN: %oss-O2-compile-and-run
// UNSUPPORTED: true
/*
<testinfo>
test_generator="config/mercurium-ompss-2"
test_nolink=yes
</testinfo>
*/

#pragma oss task
void f()
{ }

void g() {
    #pragma oss task
    {
        f();
    }

    #pragma oss task for chunksize(4)
    for (int i = 0; i < 100; ++i)
    {}

    #pragma oss taskwait
}
