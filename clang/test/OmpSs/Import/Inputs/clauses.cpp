
#pragma oss task device(cuda) ndrange(N, 1, 1)
template<int N>
void f2();

#pragma oss task device(fpga) num_instances(1337) onto(0x300000000) num_repetitions(1234) \
    period(1000) affinity(1234567) copy_in([100]i) copy_out([100]i) copy_inout([100]i) copy_deps
void f(int *i) {
    #pragma oss task if(true) final(false) cost(1) priority(2) label("a string") onready(1) \
        wait default(shared) depend(in:i[:1], { i[it], it = {0, 4, 7} })
    {}
    #pragma oss task reduction(+:[1]i)
    {}
    #pragma oss task shared(i)
    {}
    #pragma oss task private(i)
    {}
    #pragma oss task firstprivate(i)
    {}
    f2<2>();
    #pragma oss atomic update
    ++(*i);
    int n1, n2;
    #pragma oss atomic read
    n1 = n2;
    #pragma oss atomic write
    n1 = n2;
    #pragma oss atomic seq_cst
    n1 = n1+n2;
    #pragma oss atomic release
    n1 = n1+n2;
    #pragma oss atomic relaxed
    n1 = n1+n2;
    #pragma oss task for chunksize(1)
    for(int i=0;i<1;i++);
    #pragma oss taskloop grainsize(1) collapse(1)
    for(int i=0;i<1;i++);
    #pragma oss taskiter unroll(1)
    for(int i=0;i<1;i++);
}