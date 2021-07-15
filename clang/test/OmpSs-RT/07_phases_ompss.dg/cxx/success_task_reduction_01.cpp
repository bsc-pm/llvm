/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

// RUN: %oss-cxx-compile-and-run
// RUN: %oss-cxx-O2-compile-and-run

namespace N
{
    struct A
    {
        void foo();

        void bar()
        {
            int x = 0;

            #pragma oss task reduction(+: x)
            {
                x++;
            }

            #pragma oss taskwait
        }
    };

    void A::foo()
    {
        int x = 0;

        #pragma oss task reduction(+: x)
        {
            x++;
        }

        #pragma oss taskwait
    }
}

int main()
{
    N::A a;
    a.foo();
    a.bar();
}
