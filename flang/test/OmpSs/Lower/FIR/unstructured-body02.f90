! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

PROGRAM P
INTEGER :: X
INTEGER :: Y
INTEGER :: ARRAY(10, 10)

!$OSS TASK DO
DO X=0,10
DO Y=0,10
if (Y .eq. 7) EXIT
ARRAY(X,Y) = X + Y
!Y = Y + 1
END DO
END DO
!$OSS END TASK DO

END PROGRAM

! This tests checks allocas inside the directive
! and unstructured code generation.

!FIRDialect: oss.task_for
!FIRDialect: %{{.*}} = fir.alloca i32 {pinned}
!FIRDialect:   br ^[[BB1:.*]]
!FIRDialect: ^[[BB1]]:  // pred: ^bb0
!FIRDialect:   br ^[[BB2:.*]]
!FIRDialect: ^[[BB2]]:
!FIRDialect:   cond_br %{{.*}}, ^[[BB3:.*]], ^[[BB6:.*]]
!FIRDialect: ^[[BB3]]:
!FIRDialect:   cond_br %{{.*}}, ^[[BB4:.*]], ^[[BB5:.*]]
!FIRDialect: ^[[BB4]]:
!FIRDialect:   br ^[[BB6:.*]]
!FIRDialect: ^[[BB5]]:
!FIRDialect:   br ^bb2
!FIRDialect: ^[[BB6]]:
!FIRDialect:   oss.terminator
