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

// CHECK: %0 = load [10 x [20 x %struct.S]]*, [10 x [20 x %struct.S]]** %rs, align 8
// CHECK-NEXT: %1 = load [10 x [20 x %struct.S]]*, [10 x [20 x %struct.S]]** %rs1, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]* %0), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]* %0, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %0, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]* %1), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]* %1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]* %s), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]* %s, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"([10 x [20 x %struct.S]]* %s1), "QUAL.OSS.COPY"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]

// CHECK: %3 = load [10 x [20 x %struct.S]]*, [10 x [20 x %struct.S]]** %rs, align 8
// CHECK-NEXT: %4 = load [10 x [20 x %struct.S]]*, [10 x [20 x %struct.S]]** %rs1, align 8
// CHECK-NEXT: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]* %3), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]* %3, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %3, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]* %4), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]* %4, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %4, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]* %s), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]* %s, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"([10 x [20 x %struct.S]]* %s1), "QUAL.OSS.INIT"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"([10 x [20 x %struct.S]]* %s1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]

// CHECK: define internal void @oss_copy_ctor_ZN1SC1ERS_i(%struct.S* noundef %0, %struct.S* noundef %1, i64 noundef %2)
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
// CHECK-NEXT:   call void @_ZN1SC1ERS_i(%struct.S* noundef nonnull align 4 dereferenceable(4) %arrayctor.dst.cur, %struct.S* noundef nonnull align 4 dereferenceable(4) %arrayctor.src.cur, i32 noundef{{( signext)?}} 0)
// CHECK-NEXT:   %arrayctor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.dst.cur, i64 1
// CHECK-NEXT:   %arrayctor.src.next = getelementptr inbounds %struct.S, %struct.S* %arrayctor.src.cur, i64 1
// CHECK-NEXT:   %arrayctor.done = icmp eq %struct.S* %arrayctor.dst.next, %arrayctor.dst.end
// CHECK-NEXT:   br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop
// CHECK: arrayctor.cont:                                   ; preds = %arrayctor.loop
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @oss_dtor_ZN1SD1Ev(%struct.S* noundef %0, i64 noundef %1)
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
// CHECK-NEXT:   call void @_ZN1SD1Ev(%struct.S* noundef nonnull align 4 dereferenceable(4) %arraydtor.dst.cur)
// CHECK-NEXT:   %arraydtor.dst.next = getelementptr inbounds %struct.S, %struct.S* %arraydtor.dst.cur, i64 1
// CHECK-NEXT:   %arraydtor.done = icmp eq %struct.S* %arraydtor.dst.next, %arraydtor.dst.end
// CHECK-NEXT:   br i1 %arraydtor.done, label %arraydtor.cont, label %arraydtor.loop
// CHECK: arraydtor.cont:                                   ; preds = %arraydtor.loop
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @oss_ctor_ZN1SC1Ev(%struct.S* noundef %0, i64 noundef %1)
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
// CHECK-NEXT:   call void @_ZN1SC1Ev(%struct.S* noundef nonnull align 4 dereferenceable(4) %arrayctor.dst.cur)
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

// CHECK: %0 = load %struct.S*, %struct.S** %rx, align 8
// CHECK-NEXT: %1 = load %struct.S*, %struct.S** %rx1, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %0), "QUAL.OSS.COPY"(%struct.S* %0, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S* %0, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %1), "QUAL.OSS.COPY"(%struct.S* %1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S* %1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %x), "QUAL.OSS.COPY"(%struct.S* %x, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S* %x, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.FIRSTPRIVATE"(%struct.S* %x1), "QUAL.OSS.COPY"(%struct.S* %x1, void (%struct.S*, %struct.S*, i64)* @oss_copy_ctor_ZN1SC1ERS_i), "QUAL.OSS.DEINIT"(%struct.S* %x1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]

// CHECK: %3 = load %struct.S*, %struct.S** %rx, align 8
// CHECK-NEXT: %4 = load %struct.S*, %struct.S** %rx1, align 8
// CHECK-NEXT: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(%struct.S* %3), "QUAL.OSS.INIT"(%struct.S* %3, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %3, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(%struct.S* %4), "QUAL.OSS.INIT"(%struct.S* %4, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %4, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(%struct.S* %x), "QUAL.OSS.INIT"(%struct.S* %x, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %x, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev), "QUAL.OSS.PRIVATE"(%struct.S* %x1), "QUAL.OSS.INIT"(%struct.S* %x1, void (%struct.S*, i64)* @oss_ctor_ZN1SC1Ev), "QUAL.OSS.DEINIT"(%struct.S* %x1, void (%struct.S*, i64)* @oss_dtor_ZN1SD1Ev) ]

S s;
S &rs = s;
extern S &res;
void global_ref() {
    #pragma oss task in(res.x, rs.x)
    {}
}

// CHECK: %0 = load %struct.S*, %struct.S** @res, align 8
// CHECK-NEXT: %1 = load %struct.S*, %struct.S** @rs, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %0), "QUAL.OSS.SHARED"(%struct.S* %1), "QUAL.OSS.DEP.IN"(%struct.S* %0, [6 x i8] c"res.x\00", %struct._depend_unpack_t (%struct.S*)* @compute_dep, %struct.S* %0), "QUAL.OSS.DEP.IN"(%struct.S* %1, [5 x i8] c"rs.x\00", %struct._depend_unpack_t.0 (%struct.S*)* @compute_dep.1, %struct.S* %1) ]

// CHECK: define internal %struct._depend_unpack_t @compute_dep(%struct.S* %res)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %x = getelementptr inbounds %struct.S, %struct.S* %res, i32 0, i32 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(%struct.S* %rs)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %x = getelementptr inbounds %struct.S, %struct.S* %rs, i32 0, i32 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %4
// CHECK-NEXT: }
