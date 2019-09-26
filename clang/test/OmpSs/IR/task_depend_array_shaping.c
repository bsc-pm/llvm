// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int array[10][20];
void foo1(int **p, int n) {
    #pragma oss task depend(in : [n + 1]p, [n + 2]array)
    {}
    #pragma oss task depend(in : ([3]p)[2], ([4]array)[3])
    {}
    #pragma oss task depend(in : ([3]p)[2 : n], ([4]array)[3 : n])
    {}
    #pragma oss task depend(in : [3]p[2], [4]array[3])
    {}
}

// CHECK:   %0 = load i32**, i32*** %p.addr, align 8, !dbg !9
// CHECK-NEXT:   %1 = load i32, i32* %n.addr, align 4, !dbg !9
// CHECK-NEXT:   %add = add nsw i32 %1, 1, !dbg !9
// CHECK-NEXT:   %2 = zext i32 %add to i64, !dbg !9
// CHECK-NEXT:   %3 = mul i64 %2, 8, !dbg !9
// CHECK-NEXT:   %4 = mul i64 %2, 8, !dbg !9
// CHECK-NEXT:   %5 = load i32, i32* %n.addr, align 4, !dbg !9
// CHECK-NEXT:   %add1 = add nsw i32 %5, 2, !dbg !9
// CHECK-NEXT:   %6 = zext i32 %add1 to i64, !dbg !9
// CHECK-NEXT:   %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.DEP.IN"(i32** %0, i64 %3, i64 0, i64 %4), "QUAL.OSS.DEP.IN"([20 x i32]* getelementptr inbounds ([10 x [20 x i32]], [10 x [20 x i32]]* @array, i64 0, i64 0), i64 80, i64 0, i64 80, i64 %6, i64 0, i64 %6) ], !dbg !9
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %7), !dbg !10
// CHECK-NEXT:   %8 = load i32**, i32*** %p.addr, align 8, !dbg !11
// CHECK-NEXT:   %9 = bitcast i32** %8 to [3 x i32*]*, !dbg !11
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [3 x i32*], [3 x i32*]* %9, i64 0, i64 0, !dbg !11
// CHECK-NEXT:   %10 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.DEP.IN"(i32** %arraydecay, i64 24, i64 16, i64 24), "QUAL.OSS.DEP.IN"([20 x i32]* getelementptr inbounds ([10 x [20 x i32]], [10 x [20 x i32]]* @array, i64 0, i64 0), i64 80, i64 0, i64 80, i64 4, i64 3, i64 4) ], !dbg !11
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %10), !dbg !12

// CHECK:  %11 = load i32, i32* %n.addr, align 4, !dbg !13
// CHECK-NEXT:  %12 = sext i32 %11 to i64, !dbg !13
// CHECK-NEXT:  %13 = add i64 2, %12, !dbg !13
// CHECK-NEXT:  %14 = load i32**, i32*** %p.addr, align 8, !dbg !13
// CHECK-NEXT:  %15 = bitcast i32** %14 to [3 x i32*]*, !dbg !13
// CHECK-NEXT:  %arraydecay2 = getelementptr inbounds [3 x i32*], [3 x i32*]* %15, i64 0, i64 0, !dbg !13
// CHECK-NEXT:  %16 = mul i64 %13, 8, !dbg !13
// CHECK-NEXT:  %17 = load i32, i32* %n.addr, align 4, !dbg !13
// CHECK-NEXT:  %18 = sext i32 %17 to i64, !dbg !13
// CHECK-NEXT:  %19 = add i64 3, %18, !dbg !13
// CHECK-NEXT:  %20 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.FIRSTPRIVATE"(i32* %n.addr), "QUAL.OSS.DEP.IN"(i32** %arraydecay2, i64 24, i64 16, i64 %16), "QUAL.OSS.DEP.IN"([20 x i32]* getelementptr inbounds ([10 x [20 x i32]], [10 x [20 x i32]]* @array, i64 0, i64 0), i64 80, i64 0, i64 80, i64 4, i64 3, i64 %19) ], !dbg !13
// CHECK-NEXT:  call void @llvm.directive.region.exit(token %20), !dbg !14

// CHECK:  %21 = load i32**, i32*** %p.addr, align 8, !dbg !15
// CHECK-NEXT:  %arrayidx = getelementptr inbounds i32*, i32** %21, i64 2, !dbg !15
// CHECK-NEXT:  %22 = load i32*, i32** %arrayidx, align 8, !dbg !15
// CHECK-NEXT:  %23 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"([10 x [20 x i32]]* @array), "QUAL.OSS.FIRSTPRIVATE"(i32*** %p.addr), "QUAL.OSS.DEP.IN"(i32* %22, i64 12, i64 0, i64 12), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds ([10 x [20 x i32]], [10 x [20 x i32]]* @array, i64 0, i64 3, i64 0), i64 16, i64 0, i64 16) ], !dbg !15
// CHECK-NEXT:  call void @llvm.directive.region.exit(token %23), !dbg !16
