! Test test checks we do not emit duplicates data-sharings

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program task
    INTEGER :: I, J, K

    !$OSS TASK DEFAULT(FIRSTPRIVATE) PRIVATE(J, J) FIRSTPRIVATE(I, K, K)
    I = 1
    !$OSS END TASK

end program

!FIRDialect: oss.task private(%1 : !fir.ref<i32>) firstprivate(%0, %2 : !fir.ref<i32>, !fir.ref<i32>) default( deffirstprivate)

