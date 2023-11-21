! This test checks lowering of OmpSs-2 DepOp.

! Test for subroutine

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
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

!FIRDialect: %[[VAR_X_LOAD:.*]] = fir.load %arg0 : !fir.ref<i32>
!FIRDialect: %{{.*}} = arith.constant 6 : i32
!FIRDialect: %[[X_PLUS_6:.*]] = arith.addi %[[VAR_X_LOAD]], %{{.*}} : i32
!FIRDialect: %[[X_PLUS_6_64:.*]] = fir.convert %[[X_PLUS_6]] : (i32) -> i64
!FIRDialect: %[[DEP_0:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep0) arguments(%[[X_PLUS_6_64]], %[[VAR_ARRAY2]] : i64, !fir.ref<!fir.array<?xi32>>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: shared(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_0]] : i32)

!FIRDialect: %[[DEP_1:.*]] = oss.dependency base(%arg1 : !fir.ref<!fir.array<?xi32>>) function(@compute.dep1) arguments(%arg1, %[[VAR_I]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME: shared(%arg1 : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_1]] : i32)

!FIRDialect: %[[VAR_X_LOAD_0:.*]] = fir.load %arg0 : !fir.ref<i32>
!FIRDialect: %{{.*}} = arith.constant 6 : i32
!FIRDialect: %[[X_PLUS_6_0:.*]] = arith.addi %[[VAR_X_LOAD_0]], %{{.*}} : i32
!FIRDialect: %[[X_PLUS_6_64_0:.*]] = fir.convert %[[X_PLUS_6_0]] : (i32) -> i64
!FIRDialect: %[[DEP_2:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep2) arguments(%[[X_PLUS_6_64_0]], %[[VAR_ARRAY2]], %[[VAR_I]] : i64, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME: shared(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_2]] : i32)

!FIRDialect: %[[DEP_3:.*]] = oss.dependency base(%arg1 : !fir.ref<!fir.array<?xi32>>) function(@compute.dep3) arguments(%arg1, %[[VAR_I]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect-SAME: shared(%arg1 : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_3]] : i32)

!FIRDialect: %[[VAR_X_LOAD_1:.*]] = fir.load %arg0 : !fir.ref<i32>
!FIRDialect: %{{.*}} = arith.constant 6 : i32
!FIRDialect: %[[X_PLUS_6_1:.*]] = arith.addi %[[VAR_X_LOAD_1]], %{{.*}} : i32
!FIRDialect: %[[X_PLUS_6_64_1:.*]] = fir.convert %[[X_PLUS_6_1]] : (i32) -> i64
!FIRDialect: %[[DEP_4:.*]] = oss.dependency base(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep4) arguments(%[[X_PLUS_6_64_1]], %[[VAR_ARRAY2]] : i64, !fir.ref<!fir.array<?xi32>>) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: shared(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?xi32>>)
!FIRDialect-SAME: out(%[[DEP_4]] : i32)

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: i64, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c4_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.convert %arg0 : (i64) -> index
!FIRDialect-NEXT:   %2 = arith.subi %1, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %3 = arith.addi %2, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %4 = arith.cmpi sgt, %3, %c0 : index
!FIRDialect-NEXT:   %5 = arith.select %4, %3, %c0 : index
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_0 = arith.constant 4 : i64
!FIRDialect-NEXT:   %6 = arith.addi %c1_i64, %arg0 : i64
!FIRDialect-NEXT:   %7 = arith.subi %6, %c4_i64_0 : i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_2 = arith.constant 4 : i64
!FIRDialect-NEXT:   %8 = arith.subi %c4_i64_0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %9 = arith.subi %arg0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %10 = arith.addi %9, %c1_i64_1 : i64
!FIRDialect-NEXT:   %11 = arith.muli %7, %c4_i64_2 : i64
!FIRDialect-NEXT:   %12 = arith.muli %8, %c4_i64_2 : i64
!FIRDialect-NEXT:   %13 = arith.muli %10, %c4_i64_2 : i64
!FIRDialect-NEXT:   return %arg1, %11, %12, %13 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

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

!FIRDialect-LABEL: func.func @compute.dep2(%arg0: i64, %arg1: !fir.ref<!fir.array<?xi32>>, %arg2: !fir.ref<i32>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c4_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.convert %arg0 : (i64) -> index
!FIRDialect-NEXT:   %2 = arith.subi %1, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %3 = arith.addi %2, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %4 = arith.cmpi sgt, %3, %c0 : index
!FIRDialect-NEXT:   %5 = arith.select %4, %3, %c0 : index
!FIRDialect-NEXT:   %c4_i64_0 = arith.constant 4 : i64
!FIRDialect-NEXT:   %6 = arith.subi %arg0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %7 = arith.addi %6, %c1_i64 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %8 = arith.cmpi sgt, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %9 = arith.select %8, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %10 = fir.load %arg2 : !fir.ref<i32>
!FIRDialect-NEXT:   %11 = fir.convert %10 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_2 = arith.constant 4 : i64
!FIRDialect-NEXT:   %12 = arith.subi %11, %c4_i64_0 : i64
!FIRDialect-NEXT:   %13 = arith.subi %11, %c4_i64_0 : i64
!FIRDialect-NEXT:   %14 = arith.addi %13, %c1_i64_1 : i64
!FIRDialect-NEXT:   %15 = arith.muli %9, %c4_i64_2 : i64
!FIRDialect-NEXT:   %16 = arith.muli %12, %c4_i64_2 : i64
!FIRDialect-NEXT:   %17 = arith.muli %14, %c4_i64_2 : i64
!FIRDialect-NEXT:   return %arg1, %15, %16, %17 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

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

!FIRDialect-LABEL: func.func @compute.dep4(%arg0: i64, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c4_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.convert %arg0 : (i64) -> index
!FIRDialect-NEXT:   %2 = arith.subi %1, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %3 = arith.addi %2, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %4 = arith.cmpi sgt, %3, %c0 : index
!FIRDialect-NEXT:   %5 = arith.select %4, %3, %c0 : index
!FIRDialect-NEXT:   %c4_i64_0 = arith.constant 4 : i64
!FIRDialect-NEXT:   %6 = arith.subi %arg0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %7 = arith.addi %6, %c1_i64 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %8 = arith.cmpi sgt, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %9 = arith.select %8, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64_2 = arith.constant 4 : i64
!FIRDialect-NEXT:   %10 = arith.subi %c4_i64_0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %11 = arith.subi %arg0, %c4_i64_0 : i64
!FIRDialect-NEXT:   %12 = arith.addi %11, %c1_i64_1 : i64
!FIRDialect-NEXT:   %13 = arith.muli %9, %c4_i64_2 : i64
!FIRDialect-NEXT:   %14 = arith.muli %10, %c4_i64_2 : i64
!FIRDialect-NEXT:   %15 = arith.muli %12, %c4_i64_2 : i64
!FIRDialect-NEXT:   return %arg1, %13, %14, %15 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64

