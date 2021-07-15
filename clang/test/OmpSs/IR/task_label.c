// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task label(s)
void bar(const char *s);

const char text[] = "T2";
void foo() {
    #pragma oss task label("T1")
    {}
    #pragma oss task label(text)
    {}
    bar("T3");
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.LABEL"(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0)) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.LABEL"(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @text, i64 0, i64 0)) ]

// CHECK: store i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i64 0, i64 0), i8** %call_arg, align 8
// CHECK-NEXT: %2 = load i8*, i8** %call_arg, align 8
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i8** %call_arg), "QUAL.OSS.LABEL"(i8* %2), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"bar:4:9\00") ]

