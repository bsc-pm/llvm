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


!FIRDialect-LABEL: func.func @compute.dep0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c4_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:   %c6_i32 = arith.constant 6 : i32
!FIRDialect-NEXT:   %2 = arith.addi %1, %c6_i32 : i32
!FIRDialect-NEXT:   %3 = fir.convert %2 : (i32) -> i64
!FIRDialect-NEXT:   %4 = fir.convert %3 : (i64) -> index
!FIRDialect-NEXT:   %5 = arith.subi %4, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %6 = arith.addi %5, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %7 = arith.cmpi sgt, %6, %c0 : index
!FIRDialect-NEXT:   %8 = arith.select %7, %6, %c0 : index
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_0 = arith.constant 4 : i64
!FIRDialect-NEXT:   %9 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:   %c6_i32_1 = arith.constant 6 : i32
!FIRDialect-NEXT:   %10 = arith.addi %9, %c6_i32_1 : i32
!FIRDialect-NEXT:   %11 = fir.convert %10 : (i32) -> i64
!FIRDialect-NEXT:   %12 = arith.addi %c1_i64, %11 : i64
!FIRDialect-NEXT:   %13 = arith.subi %12, %c4_i64_0 : i64
!FIRDialect-NEXT:   %c1_i64_2 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_3 = arith.constant 4 : i64
!FIRDialect-NEXT:   %14 = arith.subi %c4_i64_0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %15 = arith.subi %11, %c4_i64_0 : i64
!FIRDialect-NEXT:   %16 = arith.addi %15, %c1_i64_2 : i64
!FIRDialect-NEXT:   %17 = arith.muli %13, %c4_i64_3 : i64
!FIRDialect-NEXT:   %18 = arith.muli %14, %c4_i64_3 : i64
!FIRDialect-NEXT:   %19 = arith.muli %16, %c4_i64_3 : i64
!FIRDialect-NEXT:   return %arg1, %17, %18, %19 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep1(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.undefined index
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %1 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:   %2 = fir.convert %1 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %3 = arith.subi %2, %c1_i64 : i64
!FIRDialect-NEXT:   %4 = arith.subi %2, %c1_i64 : i64
!FIRDialect-NEXT:   %5 = arith.addi %4, %c1_i64_0 : i64
!FIRDialect-NEXT:   %6 = arith.muli %c1_i64, %c4_i64 : i64
!FIRDialect-NEXT:   %7 = arith.muli %3, %c4_i64 : i64
!FIRDialect-NEXT:   %8 = arith.muli %5, %c4_i64 : i64
!FIRDialect-NEXT:   return %arg0, %6, %7, %8 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep2(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<?xi32>>, %arg2: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c4_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:   %c6_i32 = arith.constant 6 : i32
!FIRDialect-NEXT:   %2 = arith.addi %1, %c6_i32 : i32
!FIRDialect-NEXT:   %3 = fir.convert %2 : (i32) -> i64
!FIRDialect-NEXT:   %4 = fir.convert %3 : (i64) -> index
!FIRDialect-NEXT:   %5 = arith.subi %4, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %6 = arith.addi %5, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %7 = arith.cmpi sgt, %6, %c0 : index
!FIRDialect-NEXT:   %8 = arith.select %7, %6, %c0 : index
!FIRDialect-NEXT:   %c4_i64_0 = arith.constant 4 : i64
!FIRDialect-NEXT:   %9 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:   %c6_i32_1 = arith.constant 6 : i32
!FIRDialect-NEXT:   %10 = arith.addi %9, %c6_i32_1 : i32
!FIRDialect-NEXT:   %11 = fir.convert %10 : (i32) -> i64
!FIRDialect-NEXT:   %12 = arith.subi %11, %c4_i64_0 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %13 = arith.addi %12, %c1_i64 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %14 = arith.cmpi sgt, %13, %c0_i64 : i64
!FIRDialect-NEXT:   %15 = arith.select %14, %13, %c0_i64 : i64
!FIRDialect-NEXT:   %16 = fir.load %arg2 : !fir.ref<i32>
!FIRDialect-NEXT:   %17 = fir.convert %16 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_2 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_3 = arith.constant 4 : i64
!FIRDialect-NEXT:   %18 = arith.subi %17, %c4_i64_0 : i64
!FIRDialect-NEXT:   %19 = arith.subi %17, %c4_i64_0 : i64
!FIRDialect-NEXT:   %20 = arith.addi %19, %c1_i64_2 : i64
!FIRDialect-NEXT:   %21 = arith.muli %15, %c4_i64_3 : i64
!FIRDialect-NEXT:   %22 = arith.muli %18, %c4_i64_3 : i64
!FIRDialect-NEXT:   %23 = arith.muli %20, %c4_i64_3 : i64
!FIRDialect-NEXT:   return %arg1, %21, %22, %23 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep3(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.undefined index
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %1 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:   %2 = fir.convert %1 : (i32) -> i64
!FIRDialect-NEXT:   %3 = fir.load %arg1 : !fir.ref<i32>
!FIRDialect-NEXT:   %c10_i32 = arith.constant 10 : i32
!FIRDialect-NEXT:   %4 = arith.addi %3, %c10_i32 : i32
!FIRDialect-NEXT:   %5 = fir.convert %4 : (i32) -> i64
!FIRDialect-NEXT:   %6 = arith.subi %5, %2 : i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %7 = arith.addi %6, %c1_i64_0 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %8 = arith.cmpi sgt, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %9 = arith.select %8, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %10 = arith.subi %2, %c1_i64 : i64
!FIRDialect-NEXT:   %11 = arith.subi %5, %c1_i64 : i64
!FIRDialect-NEXT:   %12 = arith.addi %11, %c1_i64_1 : i64
!FIRDialect-NEXT:   %13 = arith.muli %9, %c4_i64 : i64
!FIRDialect-NEXT:   %14 = arith.muli %10, %c4_i64 : i64
!FIRDialect-NEXT:   %15 = arith.muli %12, %c4_i64 : i64
!FIRDialect-NEXT:   return %arg0, %13, %14, %15 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

!FIRDialect-LABEL: func.func @compute.dep4(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c4_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:   %c6_i32 = arith.constant 6 : i32
!FIRDialect-NEXT:   %2 = arith.addi %1, %c6_i32 : i32
!FIRDialect-NEXT:   %3 = fir.convert %2 : (i32) -> i64
!FIRDialect-NEXT:   %4 = fir.convert %3 : (i64) -> index
!FIRDialect-NEXT:   %5 = arith.subi %4, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %6 = arith.addi %5, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %7 = arith.cmpi sgt, %6, %c0 : index
!FIRDialect-NEXT:   %8 = arith.select %7, %6, %c0 : index
!FIRDialect-NEXT:   %c4_i64_0 = arith.constant 4 : i64
!FIRDialect-NEXT:   %9 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:   %c6_i32_1 = arith.constant 6 : i32
!FIRDialect-NEXT:   %10 = arith.addi %9, %c6_i32_1 : i32
!FIRDialect-NEXT:   %11 = fir.convert %10 : (i32) -> i64
!FIRDialect-NEXT:   %12 = arith.subi %11, %c4_i64_0 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %13 = arith.addi %12, %c1_i64 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %14 = arith.cmpi sgt, %13, %c0_i64 : i64
!FIRDialect-NEXT:   %15 = arith.select %14, %13, %c0_i64 : i64
!FIRDialect-NEXT:   %c1_i64_2 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_3 = arith.constant 4 : i64
!FIRDialect-NEXT:   %16 = arith.subi %c4_i64_0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %17 = arith.subi %11, %c4_i64_0 : i64
!FIRDialect-NEXT:   %18 = arith.addi %17, %c1_i64_2 : i64
!FIRDialect-NEXT:   %19 = arith.muli %15, %c4_i64_3 : i64
!FIRDialect-NEXT:   %20 = arith.muli %16, %c4_i64_3 : i64
!FIRDialect-NEXT:   %21 = arith.muli %18, %c4_i64_3 : i64
!FIRDialect-NEXT:   return %arg1, %19, %20, %21 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

