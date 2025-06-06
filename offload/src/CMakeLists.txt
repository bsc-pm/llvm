message(STATUS "Building offloading runtime library libomptarget.")

if(LIBOMP_STANDALONE)
  set(LIBOMP ${LIBOMP_STANDALONE})
else()
  set(LIBOMP omp)
endif()

add_llvm_library(omptarget
  SHARED

  device.cpp
  interface.cpp
  omptarget.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp

  KernelLanguage/API.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  Object

  LINK_LIBS
  PUBLIC
  ${LIBOMP}

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(omptarget PRIVATE
  TARGET_NAME=omptarget
  DEBUG_PREFIX="omptarget"
)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  target_link_libraries(omptarget PRIVATE omptarget.rtl.${plugin})
endforeach()

target_compile_options(omptarget PRIVATE ${offload_compile_flags})
target_link_options(omptarget PRIVATE ${offload_link_flags})

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

if (NOSV_PKG_FOUND)
if(LIBOMPV_STANDALONE)
  set(LIBOMPV ${LIBOMPV_STANDALONE})
else()
  set(LIBOMPV ompv)
endif()

add_llvm_library(ompvtarget
  SHARED

  device.cpp
  interface.cpp
  omptarget.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp

  KernelLanguage/API.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  Object

  LINK_LIBS
  PUBLIC
  ${LIBOMPV}

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(ompvtarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(ompvtarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(ompvtarget PRIVATE
  TARGET_NAME=ompvtarget
  DEBUG_PREFIX="ompvtarget"
)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  target_link_libraries(ompvtarget PRIVATE omptarget.rtl.${plugin})
endforeach()

target_compile_options(ompvtarget PRIVATE ${offload_compile_flags})
target_compile_definitions(ompvtarget PUBLIC -DKMP_OMPV_ENABLED)
target_link_options(ompvtarget PRIVATE ${offload_link_flags})

# libompvtarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(ompvtarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS ompvtarget LIBRARY COMPONENT ompvtarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
endif() # NOSV_PKG_FOUND

