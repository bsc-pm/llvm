; RUN: opt %s -passes=ompss-2-pre -S | FileCheck %s --check-prefix=PRE
; RUN: opt %s -passes='ompss-2-pre,ompss-2' -S | FileCheck %s --check-prefix=XFORM
; ModuleID = 'unreachable_patch_infinite_loop.ll'
source_filename = "unreachable_patch_infinite_loop.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; C:
;   int main(void) {
;     #pragma oss task
;     for (;;);
;     return 0;
;   }
;
; The task body is an infinite loop, so the matching llvm.directive.region.exit
; intrinsic ends up in a block with no predecessors. ompss-2-pre must keep
; that block alive (otherwise the entry/exit pair is broken and the analysis
; would assert), and the full ompss-2 transform must lower the directive
; without crashing.

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  br label %for.cond

dead:                                             ; No predecessors!
  call void @llvm.directive.region.exit(token %0)
  ret i32 0
}

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

; ompss-2-pre re-routes the unreachable `dead` block as a (runtime-unreachable)
; case of a synthetic switch in `entry` (later folded to an icmp/cond-br by
; ConstantFoldTerminator inside removeUnreachableBlocks). The
; directive.region.exit therefore survives the cleanup and the entry/exit
; pair stays linked.
;
; PRE-LABEL: define dso_local i32 @main
; PRE:       entry:
; PRE:         [[TOKEN:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
; PRE:         [[ANCHOR:%.*]] = freeze i32 undef
; PRE-NEXT:    [[COND:%.*]] = icmp eq i32 [[ANCHOR]], 1
; PRE-NEXT:    br i1 [[COND]], label %[[EXITBB:.*]], label %for.cond
; PRE:       for.cond:
; PRE-NEXT:    br label %for.cond
; PRE:       [[EXITBB]]:
; PRE-NEXT:    call void @llvm.directive.region.exit(token [[TOKEN]])
; PRE-NEXT:    ret i32 0

; The full transform pass must lower the task: a real nanos6_create_task call
; appears and the directive.region.entry/exit intrinsics are gone from @main.
;
; XFORM-LABEL: define dso_local i32 @main
; XFORM:         call void @nanos6_create_task(
; XFORM-NOT:     call {{.*}} @llvm.directive.region.entry
; XFORM-NOT:     call {{.*}} @llvm.directive.region.exit
