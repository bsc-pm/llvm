! This test checks lowering of OmpSs-2 VlaDimOp.

! Test for subroutine

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

subroutine task(X)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY1(1: x + 6,1:3)
    INTEGER :: ARRAY2(1: x + 6,2:3)
    INTEGER :: ARRAY3(4: x + 6,1:3)
    INTEGER :: ARRAY4(4: x + 6,2:3)

    INTEGER :: I
    INTEGER :: J

    !$OSS TASK DO
    DO J=1,3
      DO I=1,x+6
        ARRAY1(I,J) = (I+J)
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=2,3
      DO I=1,x+6
        ARRAY2(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=1,3
      DO I=4,x+6
        ARRAY3(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=2,3
      DO I=4,x+6
        ARRAY4(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

end subroutine

!FIRDialect-LABEL: func @_QPtask(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) {
!FIRDialect: %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskEi"}
!FIRDialect-NEXT: %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtaskEj"}

!FIRDialect: %{{.*}} = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT: %{{.*}} = arith.constant 6 : i32
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY1_E1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 3 : i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY1_E2:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY1:.*]] = fir.alloca !fir.array<?x3xi32>, %[[VAR_ARRAY1_E1]] {bindc_name = "array1", uniq_name = "_QFtaskEarray1"}

!FIRDialect: %{{.*}} = arith.constant 1 : i64
!FIRDialect-NEXT: %[[VAR_ARRAY2_LB1:.*]] = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect-NEXT: %{{.*}} = arith.constant 6 : i32
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : index
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY2_E1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 2 : i64
!FIRDialect-NEXT: %[[VAR_ARRAY2_LB2:.*]] = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.constant 3 : i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : index
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY2_E2:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY2:.*]] = fir.alloca !fir.array<?x2xi32>, %[[VAR_ARRAY2_E1]] {bindc_name = "array2", uniq_name = "_QFtaskEarray2"}

!FIRDialect: %{{.*}} = arith.constant 4 : i64
!FIRDialect-NEXT: %[[VAR_ARRAY3_LB1:.*]] = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect-NEXT: %{{.*}} = arith.constant 6 : i32
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : index
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY3_E1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : i64
!FIRDialect-NEXT: %[[VAR_ARRAY3_LB2:.*]] = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.constant 3 : i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : index
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY3_E2:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY3:.*]] = fir.alloca !fir.array<?x3xi32>, %[[VAR_ARRAY3_E1]] {bindc_name = "array3", uniq_name = "_QFtaskEarray3"}

!FIRDialect: %{{.*}} = arith.constant 4 : i64
!FIRDialect-NEXT: %[[VAR_ARRAY4_LB1:.*]] = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect-NEXT: %{{.*}} = arith.constant 6 : i32
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i32) -> i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : index
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY4_E1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 2 : i64
!FIRDialect-NEXT: %[[VAR_ARRAY4_LB2:.*]] = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.constant 3 : i64
!FIRDialect-NEXT: %{{.*}} = fir.convert %{{.*}} : (i64) -> index
!FIRDialect-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 1 : index
!FIRDialect-NEXT: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %{{.*}} = arith.constant 0 : index
!FIRDialect-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : index
!FIRDialect-NEXT: %[[VAR_ARRAY4_E2:.*]] = arith.select %57, %56, %c0_20 : index
!FIRDialect-NEXT: %[[VAR_ARRAY4:.*]] = fir.alloca !fir.array<?x2xi32>, %[[VAR_ARRAY4_E1]] {bindc_name = "array4", uniq_name = "_QFtaskEarray4"}

!FIRDialect: %[[VLA_1:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY1]] : !fir.ref<!fir.array<?x3xi32>>) sizes(%[[VAR_ARRAY1_E1]], %[[VAR_ARRAY1_E2]] : index, index) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: vlaDims(%[[VLA_1]] : i32)
!FIRDialect-SAME: captures(%[[VAR_ARRAY1_E1]], %[[VAR_ARRAY1_E2]] : index, index)

!FIRDialect: %[[VLA_2:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY2]] : !fir.ref<!fir.array<?x2xi32>>) sizes(%[[VAR_ARRAY2_E1]], %[[VAR_ARRAY2_E2]] : index, index) lbs(%[[VAR_ARRAY2_LB1]], %[[VAR_ARRAY2_LB2]] : index, index) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: vlaDims(%[[VLA_2]] : i32)
!FIRDialect-SAME: captures(%[[VAR_ARRAY2_E1]], %[[VAR_ARRAY2_E2]], %[[VAR_ARRAY2_LB1]], %[[VAR_ARRAY2_LB2]] : index, index, index, index)

!FIRDialect: %[[VLA_3:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY3]] : !fir.ref<!fir.array<?x3xi32>>) sizes(%[[VAR_ARRAY3_E1]], %[[VAR_ARRAY3_E2]] : index, index) lbs(%[[VAR_ARRAY3_LB1]], %[[VAR_ARRAY3_LB2]] : index, index) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: vlaDims(%[[VLA_3]] : i32)
!FIRDialect-SAME: captures(%[[VAR_ARRAY3_E1]], %[[VAR_ARRAY3_E2]], %[[VAR_ARRAY3_LB1]], %[[VAR_ARRAY3_LB2]] : index, index, index, index)

!FIRDialect: %[[VLA_4:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY4]] : !fir.ref<!fir.array<?x2xi32>>) sizes(%[[VAR_ARRAY4_E1]], %[[VAR_ARRAY4_E2]] : index, index) lbs(%[[VAR_ARRAY4_LB1]], %[[VAR_ARRAY4_LB2]] : index, index) -> i32
!FIRDialect: oss.task_for
!FIRDialect-SAME: vlaDims(%[[VLA_4]] : i32)
!FIRDialect-SAME: captures(%[[VAR_ARRAY4_E1]], %[[VAR_ARRAY4_E2]], %[[VAR_ARRAY4_LB1]], %[[VAR_ARRAY4_LB2]] : index, index, index, index)
