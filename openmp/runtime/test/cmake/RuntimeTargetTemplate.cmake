# CMakeLists.txt file for unit testing OpenMP host runtime library.
include(CheckFunctionExists)
include(CheckLibraryExists)

# Some tests use math functions
check_library_exists(m sqrt "" LIBOMP_HAVE_LIBM)
# When using libgcc, -latomic may be needed for atomics
# (but when using compiler-rt, the atomics will be built-in)
# Note: we can not check for __atomic_load because clang treats it
# as special built-in and that breaks CMake checks
check_function_exists(__atomic_load_1 LIBOMP_HAVE_BUILTIN_ATOMIC)
if(NOT LIBOMP_HAVE_BUILTIN_ATOMIC)
  check_library_exists(atomic __atomic_load_1 "" LIBOMP_HAVE_LIBATOMIC)
else()
  # not needed
  set(LIBOMP_HAVE_LIBATOMIC 0)
endif()

macro(pythonize_bool var)
  if (${var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

list(APPEND OPENMP_TEST_COMPILER_FEATURE_LIST "${LIBOMP_ARCH}")
update_test_compiler_features()

pythonize_bool(LIBOMP_USE_HWLOC)
pythonize_bool(LIBOMP_OMPT_SUPPORT)
pythonize_bool(LIBOMP_OMPT_OPTIONAL)
pythonize_bool(LIBOMP_OMPX_TASKGRAPH)
pythonize_bool(LIBOMP_HAVE_LIBM)
pythonize_bool(LIBOMP_HAVE_LIBATOMIC)
pythonize_bool(OPENMP_STANDALONE_BUILD)
pythonize_bool(OPENMP_TEST_COMPILER_HAS_OMIT_FRAME_POINTER_FLAGS)
pythonize_bool(OPENMP_TEST_COMPILER_HAS_OMP_H)

