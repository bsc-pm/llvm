! Test test checks we do not emit duplicates data-sharings

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program task
    INTEGER :: I, J, K, U

    !$OSS TASK DO DEFAULT(FIRSTPRIVATE) PRIVATE(J, J) FIRSTPRIVATE(U, K, K)
    DO I=1, 10
    U = 1
    END DO
    !$OSS END TASK DO

    !$OSS TASKLOOP DEFAULT(FIRSTPRIVATE) PRIVATE(J, J) FIRSTPRIVATE(U, K, K)
    DO I=1, 10
    U = 1
    END DO
    !$OSS END TASKLOOP

    !$OSS TASKLOOP DO DEFAULT(FIRSTPRIVATE) PRIVATE(J, J) FIRSTPRIVATE(U, K, K)
    DO I=1, 10
    U = 1
    END DO
    !$OSS END TASKLOOP DO

end program

!FIRDialect: oss.task_for lower_bound(%c1_i32 : i32) upper_bound(%c10_i32 : i32) step(%c1_i32_0 : i32) loop_type(%c1_i64 : i64) ind_var(%0 : !fir.ref<i32>) private(%0, %1 : !fir.ref<i32>, !fir.ref<i32>) firstprivate(%2, %3 : !fir.ref<i32>, !fir.ref<i32>) default( deffirstprivate)
!FIRDialect: oss.taskloop lower_bound(%c1_i32_1 : i32) upper_bound(%c10_i32_2 : i32) step(%c1_i32_3 : i32) loop_type(%c1_i64_4 : i64) ind_var(%0 : !fir.ref<i32>) private(%0, %1 : !fir.ref<i32>, !fir.ref<i32>) firstprivate(%2, %3 : !fir.ref<i32>, !fir.ref<i32>) default( deffirstprivate)
!FIRDialect: oss.taskloop_for lower_bound(%c1_i32_5 : i32) upper_bound(%c10_i32_6 : i32) step(%c1_i32_7 : i32) loop_type(%c1_i64_8 : i64) ind_var(%0 : !fir.ref<i32>) private(%0, %1 : !fir.ref<i32>, !fir.ref<i32>) firstprivate(%2, %3 : !fir.ref<i32>, !fir.ref<i32>) default( deffirstprivate)
