! NOTE: Assertions have been autogenerated by /home/rpenacob/llvm-mono/mlir/utils/generate-test-checks.py
! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass %s -o - | FileCheck %s --check-prefix=LLVMIR

PROGRAM S
  IMPLICIT NONE
  INTEGER ::I
  INTEGER ::J

  !$OSS TASK
  ASSOCIATE ( Z => I + 4 )
    J = Z
  END ASSOCIATE
  !$OSS END TASK

END PROGRAM

! LLVMIR-LABEL: define void @_QQmain() {
! LLVMIR:         %[[VAL_0:[-0-9A-Za-z._]+]] = alloca i32, i64 1, align 4
! LLVMIR:         %[[VAL_1:[-0-9A-Za-z._]+]] = alloca i32, i64 1, align 4
! LLVMIR:         %[[VAL_2:[-0-9A-Za-z._]+]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAL_1]], i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAL_0]], i32 undef) ]
! LLVMIR:         br label %[[VAL_3:[-0-9A-Za-z._]+]]
! LLVMIR:       oss.end:                                          ; preds = %[[VAL_3]]
! LLVMIR:         call void @llvm.directive.region.exit(token %[[VAL_2]])
! LLVMIR:         ret void
! LLVMIR:       oss.task.region:                                  ; preds = %[[VAL_4:[-0-9A-Za-z._]+]]
! LLVMIR:         %[[VAL_5:[-0-9A-Za-z._]+]] = alloca i32, i64 1, align 4
! LLVMIR:         %[[VAL_6:[-0-9A-Za-z._]+]] = load i32, ptr %[[VAL_1]], align 4
! LLVMIR:         %[[VAL_7:[-0-9A-Za-z._]+]] = add i32 %[[VAL_6]], 4
! LLVMIR:         store i32 %[[VAL_7]], ptr %[[VAL_5]], align 4
! LLVMIR:         %[[VAL_8:[-0-9A-Za-z._]+]] = load i32, ptr %[[VAL_5]], align 4
! LLVMIR:         store i32 %[[VAL_8]], ptr %[[VAL_0]], align 4
! LLVMIR:         br label %[[VAL_9:[-0-9A-Za-z._]+]]
! LLVMIR:       }
