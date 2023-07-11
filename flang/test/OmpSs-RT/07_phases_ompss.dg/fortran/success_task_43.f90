! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_FFLAGS="--no-copy-deps"
! </testinfo>
PROGRAM P
	IMPLICIT NONE
	INTEGER, TARGET, ALLOCATABLE :: X(:, :, :)
	INTEGER, POINTER :: PTR(:, :)
	INTEGER :: I

	ALLOCATE(X(10, 20, 30))
	X = 0

	DO I=1, 30
		PTR => X(:, :, i) 
	
		!$OSS TASK INOUT(PTR)
			PTR = i
		!$OSS END TASK
	ENDDO
	!$OSS TASKWAIT

	DO I=1, 30
		IF(ANY(X(:, :, I) /= I)) STOP 1
	ENDDO
END PROGRAM P
