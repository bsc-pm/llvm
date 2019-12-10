/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/

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
// RUN: %oss-cxx-compile-and-run
//XFAIL: *
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
