add_subdirectory(omp)

# Standalone builds can use other compiler
# than clang. Le'ts be conservative and only
# pass libompv tests in controlated environments
if (NOT ${OPENMP_STANDALONE_BUILD})
  if (NOSV_PKG_FOUND)
    add_subdirectory(ompv)
    add_subdirectory(ompv_passive)
    add_subdirectory(ompv_free_agents)
    add_subdirectory(ompv_gomp)
  endif()
endif()
