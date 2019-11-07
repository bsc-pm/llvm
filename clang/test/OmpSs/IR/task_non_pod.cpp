// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
    int x;
    S();
    S(S& s, int x = 0);
    S(const S& s, int x = 0);
    ~S();
};

using Q = S;

// In theory we are going to generate a init/deinit/copy function once

void foo() {
    Q s[10][20];
    Q (&rs)[10][20] = s;
    S s1[10][20];
    Q (&rs1)[10][20] = s1;
    #pragma oss task firstprivate(rs, rs1, s, s1)
    {}
    #pragma oss task private(rs, rs1, s, s1)
    {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]** %rs), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]** %rs, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]** %rs, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]** %rs1), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]** %rs1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]** %rs1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]* %s), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]* %s, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]* %s1), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]
// CHECK:  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]** %rs), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]** %rs, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]** %rs, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]** %rs1), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]** %rs1, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]** %rs1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]* %s), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]* %s, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]* %s1), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]

// CHECK: define internal void @oss_copy_ctor_ZN1SC1ERS_i(%struct.S* %0, %struct.S* %1, i64 %2)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %.addr = alloca %struct.S*, align 8
// CHECK-NEXT:   %.addr1 = alloca %struct.S*, align 8
// CHECK-NEXT:   %.addr2 = alloca i64, align 8
// CHECK-NEXT:   store %struct.S* %0, %struct.S** %.addr, align 8
// CHECK-NEXT:   store %struct.S* %1, %struct.S** %.addr1, align 8
// CHECK-NEXT:   store i64 %2, i64* %.addr2, align 8
// CHECK-NEXT:   %3 = load %struct.S*, %struct.S** %.addr, align 8
// CHECK-NEXT:   %4 = load %struct.S*, %struct.S** %.addr1, align 8
// CHECK-NEXT:   %5 = load i64, i64* %.addr2, align 8
// CHECK-NEXT:   %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %4, i64 %5
// CHECK-NEXT:   br label %arrayctor.loop
// CHECK: arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
// CHECK-NEXT:   %arrayctor.dst.cur = phi %struct.S* [ %4, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
// CHECK-NEXT:   %arrayctor.src.cur = phi %struct.S* [ %3, %entry ], [ %arrayctor.src.next, %arrayctor.loop ]
// CHECK-NEXT:   call void @_ZN1SC1ERS_i(%struct.S* %arrayctor.dst.cur, %struct.S* dereferenceable(4) %arrayctor.src.cur, i32 0)
// CHECK-NEXT:   %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1
// CHECK-NEXT:   %arrayctor.src.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.src.cur, i64 1
// CHECK-NEXT:   %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end
// CHECK-NEXT:   br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop
// CHECK: arrayctor.cont:                                   ; preds = %arrayctor.loop
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @oss_dtor_ZN1SD1Ev(%struct.S* %0, i64 %1)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %.addr = alloca %struct.S*, align 8
// CHECK-NEXT:   %.addr1 = alloca i64, align 8
// CHECK-NEXT:   store %struct.S* %0, %struct.S** %.addr, align 8
// CHECK-NEXT:   store i64 %1, i64* %.addr1, align 8
// CHECK-NEXT:   %2 = load %struct.S*, %struct.S** %.addr, align 8
// CHECK-NEXT:   %3 = load i64, i64* %.addr1, align 8
// CHECK-NEXT:   %arraydtor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3
// CHECK-NEXT:   br label %arraydtor.loop
// CHECK: arraydtor.loop:                                   ; preds = %arraydtor.loop, %entry
// CHECK-NEXT:   %arraydtor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arraydtor.dst.next, %arraydtor.loop ]
// CHECK-NEXT:   call void @_ZN1SD1Ev(%struct.S* %arraydtor.dst.cur)
// CHECK-NEXT:   %arraydtor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arraydtor.dst.cur, i64 1
// CHECK-NEXT:   %arraydtor.done = icmp eq %struct.S* %arraydtor.dst.next, %arraydtor.dst.end
// CHECK-NEXT:   br i1 %arraydtor.done, label %arraydtor.cont, label %arraydtor.loop
// CHECK: arraydtor.cont:                                   ; preds = %arraydtor.loop
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @oss_ctor_ZN1SC1Ev(%struct.S* %0, i64 %1)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %.addr = alloca %struct.S*, align 8
// CHECK-NEXT:   %.addr1 = alloca i64, align 8
// CHECK-NEXT:   store %struct.S* %0, %struct.S** %.addr, align 8
// CHECK-NEXT:   store i64 %1, i64* %.addr1, align 8
// CHECK-NEXT:   %2 = load %struct.S*, %struct.S** %.addr, align 8
// CHECK-NEXT:   %3 = load i64, i64* %.addr1, align 8
// CHECK-NEXT:   %arrayctor.dst.end = getelementptr inbounds %struct.S, %struct.S* %2, i64 %3
// CHECK-NEXT:   br label %arrayctor.loop
// CHECK: arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
// CHECK-NEXT:   %arrayctor.dst.cur = phi %struct.S* [ %2, %entry ], [ %arrayctor.dst.next, %arrayctor.loop ]
// CHECK-NEXT:   call void @_ZN1SC1Ev(%struct.S* %arrayctor.dst.cur)
// CHECK-NEXT:   %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1
// CHECK-NEXT:   %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end
// CHECK-NEXT:   br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop
// CHECK: arrayctor.cont:                                   ; preds = %arrayctor.loop
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

int main() {
    Q x;
    Q &rx = x;
    S x1;
    S &rx1 = x1;
    #pragma oss task firstprivate(rx, rx1, x, x1)
    {}
    #pragma oss task private(rx, rx1, x, x1)
    {}
    return 0;
}
// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(%struct.S** %rx), "QUAL.OSS.COPY"(%struct.S** %rx, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S** %rx, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"(%struct.S** %rx1), "QUAL.OSS.COPY"(%struct.S** %rx1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S** %rx1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %x), "QUAL.OSS.COPY"(%struct.S* %x, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S* %x, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %x1), "QUAL.OSS.COPY"(%struct.S* %x1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S* %x1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(%struct.S** %rx), "QUAL.OSS.INIT"(%struct.S** %rx, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S** %rx, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(%struct.S** %rx1), "QUAL.OSS.INIT"(%struct.S** %rx1, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S** %rx1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(%struct.S* %x), "QUAL.OSS.INIT"(%struct.S* %x, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %x, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(%struct.S* %x1), "QUAL.OSS.INIT"(%struct.S* %x1, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %x1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]


