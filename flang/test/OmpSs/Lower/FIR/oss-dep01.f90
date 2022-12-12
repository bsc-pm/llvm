! This test checks lowering of OmpSs-2 DepOp.

! Test for subroutine

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

! Support list
! - [x] assumed-size array
! - [x] assumed-shape array

subroutine task(X, ARRAY, ARRAY1)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(*)
    INTEGER :: ARRAY1(:)
    INTEGER :: ARRAY2(4: x + 6)

    INTEGER :: I
    INTEGER :: J

    !$OSS TASK DO DEPEND(OUT: ARRAY2)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1(I))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY2(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I : I + 10))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1(I : I + 10))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    ! TODO
    !$OSS TASK DO DEPEND(OUT: ARRAY2( : ))
    DO J=1,10
    END DO
    !$OSS END TASK DO

end subroutine

!FIRDialect-LABEL: func @_QPtask(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array1"}) {
!FIRDialect: %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskEi"}
!FIRDialect-NEXT: %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtaskEj"}
!FIRDialect: %[[VAR_ARRAY2:.*]] = fir.alloca !fir.array<?xi32>, %{{.*}} {bindc_name = "array2", uniq_name = "_QFtaskEarray2"}

!FIRDialect: %[[DEP_0:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep0) arguments(%arg0, %[[VAR_ARRAY2]] : !fir.ref<i32>, !fir.ref<!fir.array<?xi32>>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: shared(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_0]] : i32)

!FIRDialect: %[[DEP_1:.*]] = oss.dependency base(%arg1 : !fir.ref<!fir.array<?xi32>>) function(@compute.dep1) arguments(%arg1, %[[VAR_I]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME: shared(%arg1 : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_1]] : i32)

!FIRDialect: %[[DEP_2:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep2) arguments(%arg0, %[[VAR_ARRAY2]], %[[VAR_I]] : !fir.ref<i32>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME: shared(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_2]] : i32)

!FIRDialect: %[[DEP_3:.*]] = oss.dependency base(%arg1 : !fir.ref<!fir.array<?xi32>>) function(@compute.dep3) arguments(%arg1, %[[VAR_I]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME: shared(%arg1 : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_3]] : i32)

!FIRDialect: %[[DEP_4:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep4) arguments(%arg0, %[[VAR_ARRAY2]] : !fir.ref<i32>, !fir.ref<!fir.array<?xi32>>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: shared(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_4]] : i32)


!FIRDialect-LABEL: func @compute.dep0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:  %c6_i32 = arith.constant 6 : i32
!FIRDialect-NEXT:  %1 = arith.addi %0, %c6_i32 : i32
!FIRDialect-NEXT:  %2 = fir.convert %1 : (i32) -> i64
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %3 = arith.addi %c1_i64, %2 : i64
!FIRDialect-NEXT:  %4 = arith.subi %3, %c4_i64 : i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64_1 = arith.constant 4 : i64
!FIRDialect-NEXT:  %5 = arith.subi %c4_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %6 = arith.subi %2, %c4_i64 : i64
!FIRDialect-NEXT:  %7 = arith.addi %6, %c1_i64_0 : i64
!FIRDialect-NEXT:  %8 = arith.muli %4, %c4_i64_1 : i64
!FIRDialect-NEXT:  %9 = arith.muli %5, %c4_i64_1 : i64
!FIRDialect-NEXT:  %10 = arith.muli %7, %c4_i64_1 : i64
!FIRDialect-NEXT:  return %arg1, %8, %9, %10 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep1(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:  %1 = fir.convert %0 : (i32) -> i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %2 = arith.subi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %3 = arith.subi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %4 = arith.addi %3, %c1_i64_0 : i64
!FIRDialect-NEXT:  %5 = arith.muli %c1_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %6 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT:  %7 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %5, %6, %7 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL:func.func @compute.dep2(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<?xi32>>, %arg2: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:  %c6_i32 = arith.constant 6 : i32
!FIRDialect-NEXT:  %1 = arith.addi %0, %c6_i32 : i32
!FIRDialect-NEXT:  %2 = fir.convert %1 : (i32) -> i64
!FIRDialect-NEXT:  %3 = arith.subi %2, %c4_i64 : i64
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %4 = arith.addi %3, %c1_i64 : i64
!FIRDialect-NEXT:  %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:  %5 = arith.cmpi sgt, %4, %c0_i64 : i64
!FIRDialect-NEXT:  %6 = arith.select %5, %4, %c0_i64 : i64
!FIRDialect-NEXT:  %7 = fir.load %arg2 : !fir.ref<i32>
!FIRDialect-NEXT:  %8 = fir.convert %7 : (i32) -> i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64_1 = arith.constant 4 : i64
!FIRDialect-NEXT:  %9 = arith.subi %8, %c4_i64 : i64
!FIRDialect-NEXT:  %10 = arith.subi %8, %c4_i64 : i64
!FIRDialect-NEXT:  %11 = arith.addi %10, %c1_i64_0 : i64
!FIRDialect-NEXT:  %12 = arith.muli %6, %c4_i64_1 : i64
!FIRDialect-NEXT:  %13 = arith.muli %9, %c4_i64_1 : i64
!FIRDialect-NEXT:  %14 = arith.muli %11, %c4_i64_1 : i64
!FIRDialect-NEXT:  return %arg1, %12, %13, %14 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep3(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:  %1 = fir.convert %0 : (i32) -> i64
!FIRDialect-NEXT:  %2 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:  %c10_i32 = arith.constant 10 : i32
!FIRDialect-NEXT:  %3 = arith.addi %2, %c10_i32 : i32
!FIRDialect-NEXT:  %4 = fir.convert %3 : (i32) -> i64
!FIRDialect-NEXT:  %5 = arith.subi %4, %1 : i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %6 = arith.addi %5, %c1_i64_0 : i64
!FIRDialect-NEXT:  %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:  %7 = arith.cmpi sgt, %6, %c0_i64 : i64
!FIRDialect-NEXT:  %8 = arith.select %7, %6, %c0_i64 : i64
!FIRDialect-NEXT:  %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %9 = arith.subi %1, %c1_i64 : i64
!FIRDialect-NEXT:  %10 = arith.subi %4, %c1_i64 : i64
!FIRDialect-NEXT:  %11 = arith.addi %10, %c1_i64_1 : i64
!FIRDialect-NEXT:  %12 = arith.muli %8, %c4_i64 : i64
!FIRDialect-NEXT:  %13 = arith.muli %9, %c4_i64 : i64
!FIRDialect-NEXT:  %14 = arith.muli %11, %c4_i64 : i64
!FIRDialect-NEXT:  return %arg0, %12, %13, %14 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep4(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:  %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:  %0 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:  %c6_i32 = arith.constant 6 : i32
!FIRDialect-NEXT:  %1 = arith.addi %0, %c6_i32 : i32
!FIRDialect-NEXT:  %2 = fir.convert %1 : (i32) -> i64
!FIRDialect-NEXT:  %3 = arith.subi %2, %c4_i64 : i64
!FIRDialect-NEXT:  %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:  %4 = arith.addi %3, %c1_i64 : i64
!FIRDialect-NEXT:  %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:  %5 = arith.cmpi sgt, %4, %c0_i64 : i64
!FIRDialect-NEXT:  %6 = arith.select %5, %4, %c0_i64 : i64
!FIRDialect-NEXT:  %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:  %c4_i64_1 = arith.constant 4 : i64
!FIRDialect-NEXT:  %7 = arith.subi %c4_i64, %c4_i64 : i64
!FIRDialect-NEXT:  %8 = arith.subi %2, %c4_i64 : i64
!FIRDialect-NEXT:  %9 = arith.addi %8, %c1_i64_0 : i64
!FIRDialect-NEXT:  %10 = arith.muli %6, %c4_i64_1 : i64
!FIRDialect-NEXT:  %11 = arith.muli %7, %c4_i64_1 : i64
!FIRDialect-NEXT:  %12 = arith.muli %9, %c4_i64_1 : i64
!FIRDialect-NEXT:  return %arg1, %10, %11, %12 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

