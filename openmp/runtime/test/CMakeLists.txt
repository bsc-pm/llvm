add_subdirectory(omp)

if (NOSV_PKG_FOUND)
  # libompv is built without OMPX-taskgraph
  set(LIBOMP_OMPX_TASKGRAPH FALSE)
  add_subdirectory(ompv)
  add_subdirectory(ompv_passive)
  add_subdirectory(ompv_free_agents)
  add_subdirectory(ompv_gomp)
endif()
