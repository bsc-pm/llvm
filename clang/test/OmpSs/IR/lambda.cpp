// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void bar(int x) {
    auto foo = [&x]() {
      #pragma oss task in(x)
      x++;
    };
    foo();
    #pragma oss task cost([&x]() { return x; }())
    {}
}
// _ZZ3bari()
// CHECK: %0 = getelementptr inbounds %class.anon, %class.anon* %foo, i32 0, i32 0
// CHECK-NEXT: store i32* %x.addr, i32** %0, align 8
// CHECK-NEXT: call void @"_ZZ3bariENK3$_0clEv"(%class.anon* noundef nonnull align 8 dereferenceable(8) %foo)
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %x.addr), "QUAL.OSS.COST"(i32 (i32*)* @compute_cost, i32* %x.addr) ]

// "_ZZ3bariENK3$_0clEv"()
// CHECK: %1 = load i32*, i32** %0, align 8
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %1), "QUAL.OSS.DEP.IN"(i32* %1, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %1) ]
// CHECK-NEXT: %3 = load i32, i32* %1, align 4
// CHECK-NEXT: %inc = add nsw i32 %3, 1
// CHECK-NEXT: store i32 %inc, i32* %1, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)

// CHECK: define internal i32 @compute_cost(i32* %x)
// CHECK: entry:
// CHECK-NEXT:   %x.addr = alloca i32*, align 8
// CHECK-NEXT:   %ref.tmp = alloca %class.anon.0, align 8
// CHECK-NEXT:   store i32* %x, i32** %x.addr, align 8
// CHECK-NEXT:   %0 = getelementptr inbounds %class.anon.0, %class.anon.0* %ref.tmp, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %call = call noundef i32 @"_ZZ3bariENK3$_1clEv"(%class.anon.0* noundef nonnull align 8 dereferenceable(8) %ref.tmp)
// CHECK-NEXT:   ret i32 %call
// CHECK-NEXT: }

// CHECK: define internal noundef i32 @"_ZZ3bariENK3$_1clEv"(%class.anon.0* noundef nonnull align 8 dereferenceable(8) %this)
// CHECK: entry:
// CHECK-NEXT:   %this.addr = alloca %class.anon.0*, align 8
// CHECK-NEXT:   store %class.anon.0* %this, %class.anon.0** %this.addr, align 8
// CHECK-NEXT:   %this1 = load %class.anon.0*, %class.anon.0** %this.addr, align 8
// CHECK-NEXT:   %0 = getelementptr inbounds %class.anon.0, %class.anon.0* %this1, i32 0, i32 0
// CHECK-NEXT:   %1 = load i32*, i32** %0, align 8
// CHECK-NEXT:   %2 = load i32, i32* %1, align 4
// CHECK-NEXT:   ret i32 %2
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %x)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
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
