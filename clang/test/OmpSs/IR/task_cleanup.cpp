// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct Foo {
    ~Foo() { }
};

void foo()
{
    #pragma oss task
    {
        int n;
        while (1) {
                Foo f;
            goto l;
        }
l:      n++; // to put something in the label
    }
    #pragma oss taskwait
}

// CHECK:   %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT:   %n = alloca i32, align 4
// CHECK-NEXT:   %f = alloca %struct.Foo, align 1
// CHECK-NEXT:   %cleanup.dest.slot = alloca i32, align 4
// CHECK-NEXT:   br label %while.cond
// CHECK: while.cond:                                       ; preds = %entry
// CHECK-NEXT:   br label %while.body
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK-NEXT:   store i32 4, i32* %cleanup.dest.slot, align 4
// CHECK-NEXT:   call void @_ZN3FooD1Ev(%struct.Foo* noundef nonnull align 1 dereferenceable(1) %f)
// CHECK-NEXT:   %cleanup.dest = load i32, i32* %cleanup.dest.slot, align 4
// CHECK-NEXT:   switch i32 %cleanup.dest, label %unreachable [
// CHECK-NEXT:     i32 4, label %l
// CHECK-NEXT:   ]
// CHECK: l:                                                ; preds = %while.body
// CHECK-NEXT:   %1 = load i32, i32* %n, align 4
// CHECK-NEXT:   %inc = add nsw i32 %1, 1
// CHECK-NEXT:   store i32 %inc, i32* %n, align 4
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT:   %2 = call i1 @llvm.directive.marker() [ "DIR.OSS"([9 x i8] c"TASKWAIT\00") ]
// CHECK-NEXT:   ret void
// CHECK: unreachable:                                      ; preds = %while.body
// CHECK-NEXT:   unreachable

