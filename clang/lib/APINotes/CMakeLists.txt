set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Support)
add_clang_library(clangAPINotes
  APINotesReader.cpp
  APINotesTypes.cpp
  APINotesWriter.cpp
  APINotesYAMLCompiler.cpp
  LINK_LIBS
    clangBasic)
