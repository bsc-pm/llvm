set(OLD_CMAKE_JOB_POOL_COMPILE CMAKE_JOB_POOL_COMPILE)
set(CMAKE_JOB_POOL_COMPILE "compile_flang")

add_flang_library(FortranParser
  Fortran-parsers.cpp
  char-buffer.cpp
  char-block.cpp
  char-set.cpp
  characters.cpp
  debug-parser.cpp
  executable-parsers.cpp
  expr-parsers.cpp
  instrumented-parser.cpp
  io-parsers.cpp
  message.cpp
  openacc-parsers.cpp
  openmp-parsers.cpp
  ompss-parsers.cpp
  parse-tree.cpp
  parsing.cpp
  preprocessor.cpp
  prescan.cpp
  program-parsers.cpp
  provenance.cpp
  source.cpp
  token-sequence.cpp
  tools.cpp
  unparse.cpp
  user-state.cpp

  LINK_LIBS
  FortranCommon

  LINK_COMPONENTS
  Support
  FrontendOpenACC

  DEPENDS
  omp_gen
  acc_gen
)

set(CMAKE_JOB_POOL_COMPILE OLD_CMAKE_JOB_POOL_COMPILE)
