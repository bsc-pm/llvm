! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

PROGRAM P
INTEGER :: X
INTEGER :: Y
INTEGER :: ARRAY(10, 10)

!$OSS TASK
DO X=0,10
DO Y=0,10
if (Y .eq. 7) EXIT
ARRAY(X,Y) = X + Y
END DO
END DO
!$OSS END TASK

END PROGRAM

! This tests checks allocas inside the directive
! and unstructured code generation.

!FIRDialect: oss.task
!FIRDialect:   %{{.*}} = fir.alloca i32 {pinned}
!FIRDialect:   %{{.*}} = fir.alloca i32 {pinned}
!FIRBuilder:   br ^[[BB1:.*]]
!FIRBuilder: ^[[BB1]]:  // 2 preds: ^bb0, ^bb7
!FIRBuilder:   cond_br %9, ^[[BB2:.*]], ^[[BB8:.*]]
!FIRBuilder: ^[[BB2]]:  // pred: ^bb1
!FIRBuilder:   br ^[[BB3:.*]]
!FIRBuilder: ^[[BB3]]:  // 2 preds: ^bb2, ^bb6
!FIRBuilder:   cond_br %14, ^[[BB4:.*]], ^[[BB7:.*]]
!FIRBuilder: ^[[BB4]]:  // pred: ^bb3
!FIRBuilder:   cond_br %16, ^[[BB5:.*]], ^[[BB6:.*]]
!FIRBuilder: ^[[BB5]]:  // pred: ^bb4
!FIRBuilder:   br ^[[BB7]]
!FIRBuilder: ^[[BB6]]:  // pred: ^bb4
!FIRBuilder:   br ^[[BB3]]
!FIRBuilder: ^[[BB7]]:  // 2 preds: ^bb3, ^bb5
!FIRBuilder:   br ^[[BB1]]
!FIRBuilder: ^[[BB8]]:  // pred: ^bb1
!FIRBuilder:   oss.terminator
