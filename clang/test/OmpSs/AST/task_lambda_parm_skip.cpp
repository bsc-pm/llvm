// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

// This test checks that lambda parameters are skipped
// from implicit dsa analysis in the task body.
// Before the fix, a deleted constructor error
// will be shown because of firstprivatizing
// 's'

struct S {
  S() = delete;
  ~S();
};

void foo(const S &s);

int main(){
  #pragma oss task //firstprivate(s)
  {
    auto l = [] (S &s){
      foo(s);
    };
  }
}

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NOT: OSSFirstprivateClause
