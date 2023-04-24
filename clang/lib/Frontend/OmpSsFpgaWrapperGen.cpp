//===- OmpssFpgaWrapperGen.cpp - Wrapper generation for OmpSs Fpga target -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements wrapper generation for OmpSs FPGA functions
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/OmpSsFgpaWrapperGen.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOmpSs.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprOmpSs.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/OmpSsFpgaTreeTransform.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#define STR_COMPONENTS_COUNT "__mcxx_taskComponents"
#define STR_OUTPORT "mcxx_outPort"
#define STR_OUTPORT_WRITE(X) "mcxx_outPort.write(" X ")"
#define STR_INPORT "mcxx_inPort"
#define STR_INSTRPORT "mcxx_instr"
#define STR_SPWNINPORT "mcxx_spawnInPort"
#define STR_SPWNINPORT_READ "mcxx_spawnInPort.read()"
#define STR_INPORT_READ "mcxx_inPort.read()"
#define STR_INPORT_READ_NODATA "mcxx_inPort.read()"
#define STR_INPORT_TYPE "hls::stream<ap_uint<64> >"
#define STR_SPWNINPORT_TYPE "hls::stream<ap_uint<8> >"
#define STR_OUTPORT_TYPE "hls::stream<mcxx_outaxis>"
#define STR_INSTRPORT_TYPE "hls::stream<__mcxx_instrData_t>"
#define STR_INPORT_DECL STR_INPORT_TYPE "& " STR_INPORT
#define STR_SPWNINPORT_DECL STR_SPWNINPORT_TYPE "& " STR_SPWNINPORT
#define STR_OUTPORT_DECL STR_OUTPORT_TYPE "& " STR_OUTPORT
#define STR_INOUTPORT_DECL STR_INPORT_DECL ", " STR_OUTPORT_DECL
#define STR_SPWNINOUTPORT_DECL STR_SPWNINPORT_DECL ", " STR_OUTPORT_DECL
#define STR_INSTRPORT_DECL STR_INSTRPORT_TYPE "& " STR_INSTRPORT
#define STR_OUTPORT_WRITE_FUN(X) "mcxx_write_out_port(" X ", " STR_OUTPORT ");"
#define STR_TASKWAIT_FUN "mcxx_taskwait(" STR_SPWNINPORT ", " STR_OUTPORT ");"
#define STR_TASK_CREATE_FUN(X) "mcxx_task_create(" X ", " STR_OUTPORT ");"
#define STR_TASKID "__mcxx_taskId"
#define STR_PARENT_TASKID "__mcxx_parent_taskId"
#define STR_FINISH_TASK_CODE "0"
#define STR_NEW_TASK_CODE "2"
#define STR_TASKWAIT_CODE "3"
#define STR_ARG_FLAG_IN_BIT "4"
#define STR_ARG_FLAG_OUT_BIT "5"

using namespace clang;

namespace {
static constexpr auto WrapperVersion = 13;

template <typename Callable>
std::optional<std::string>
getAbsoluteDirExport(const SourceManager &SourceMgr, const std::string &path,
                     const SourceLocation &Location, Callable &&Diag) {
  auto &vfs = SourceMgr.getFileManager().getVirtualFileSystem();
  if (path.empty()) {
    Diag(Location, diag::err_oss_ompss_fpga_hls_tasks_dir_missing_dir);
    return std::nullopt;
  }
  llvm::SmallVector<char, 128> realPathStr;
  if (vfs.getRealPath(path, realPathStr)) {
    Diag(Location, diag::err_oss_ompss_fpga_hls_tasks_dir_missing_dir);
    return std::nullopt;
  }
  return std::string(realPathStr.begin(), realPathStr.end());
}
template <class Callable>
bool GenerateExtractedOriginalFunction(
    Callable &&Diag, llvm::raw_ostream &outputFile, FunctionDecl *FD,
    llvm::StringRef funcName, ASTContext &sourceContext,
    SourceManager &SourceMgr, Preprocessor &PP) {
  // Dependency resolution. We use the ASTImporter utility, which is able to
  // manage any sort of C++ construct during resolution.
  DiagnosticsEngine toDiagnostics(new DiagnosticIDs(), new DiagnosticOptions());
  FileManager fileManager(SourceMgr.getFileManager().getFileSystemOpts());
  SourceManager toMgr(toDiagnostics, fileManager);
  LangOptions toLangOpts(sourceContext.getLangOpts());
  IdentifierTable toIdentifierTable;
  SelectorTable toSelectorTable;
  ASTContext toContext(toLangOpts, toMgr, toIdentifierTable, toSelectorTable,
                       PP.getBuiltinInfo(), TU_Incremental);
  toContext.InitBuiltinTypes(sourceContext.getTargetInfo());

  ASTImporter importer(toContext, toMgr.getFileManager(), sourceContext,
                       SourceMgr.getFileManager(), true);

  auto importedOrErr = importer.Import(FD);
  if (!importedOrErr) {
    auto err = importedOrErr.takeError();
    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << err;
    Diag(FD->getLocation(), diag::err_oss_fpga_dependency_analisis) << out;
    return false;
  }

  // Headers
  // We are only going to preserve the headers ending with .fpga.h or .fpga,
  // this was a restriction in the original ompss@fpga and it simplifies some
  // tasks in a major way.
  for (auto &&file : PP.getIncludedFiles()) {
    if (!file->getName().endswith(".fpga.h") &&
        !file->getName().endswith(".fpga")) {
      continue;
    }
    StringRef headerInclude;
    // We need to generate a string with two " and the path
    std::string headerIncludeStorage =
        std::string("\"").append(file->tryGetRealPathName()).append("\"");
    headerInclude = headerIncludeStorage;
    if (headerInclude.size() == 2) /*Missing path*/ {
      // This extracts the path from the source file
      auto loc = SourceMgr.getIncludeLoc(SourceMgr.translateFile(file));
      auto *data = SourceMgr.getCharacterData(loc);
      // Just to make sure I don't overrun a buffer later
      auto bufferData = SourceMgr.getBufferData(SourceMgr.getFileID(loc));
      const auto *endChar = bufferData.data() + bufferData.size();
      auto *dataEndInclude = data;
      for (; dataEndInclude < endChar && *dataEndInclude != '\n' &&
             *dataEndInclude != '\r' && *dataEndInclude != '\0';
           ++dataEndInclude)
        ;

      headerInclude = StringRef(data, dataEndInclude - data);
    }
    outputFile << "#include " << headerInclude << '\n';
  }

  // Body functions
  std::string out;
  llvm::raw_string_ostream stream(out);
  for (Decl *otherDecl : toContext.getTranslationUnitDecl()->decls()) {
    if (SourceMgr.getFileID(otherDecl->getSourceRange().getBegin()) !=
        SourceMgr.getFileID(FD->getSourceRange().getBegin())) {
      // Skip dependency not originating in the file.
      continue;
    }

    if (dyn_cast<FunctionDecl>(otherDecl) && !otherDecl->hasBody()) {
      Diag(otherDecl->getLocation(),
           diag::err_oss_fpga_missing_body_for_function_depended_by_kernel);
      Diag(FD->getLocation(), diag::note_oss_fpga_kernel);
      return false;
    }

    otherDecl->print(stream, 0, true);
    if (!out.empty() && out[out.size() - 1] != '\n') {
      stream << ";\n"; // This may lead to superfluous ';', but I'm not
                       // quite sure how to prevent them easily.
                       // Trailing ; are legal fortunately!
    }
  }
  outputFile << out;

  return true;
}

template <typename Callable> class WrapperGenerator {
  // Construction
  Callable Diag;
  std::string OutputStrHeaders;
  llvm::raw_string_ostream OutputHeaders;
  std::string OutputStr;
  llvm::raw_string_ostream Output;
  llvm::raw_ostream &OutputFinalFile;

  llvm::StringRef OrigFuncName;
  std::string TaskFuncName;

  Preprocessor &PP;
  CompilerInstance &CI;
  ReplacementMap ReplacementMap;
  PrintingPolicy printPol;

  FunctionDecl *OriginalFD;
  ASTContext &OriginalContext;
  SourceManager &OriginalSourceMgr;

  DiagnosticsEngine ToDiagnosticsEngine;
  FileManager ToFileManager;
  SourceManager ToSourceManager;
  LangOptions ToLangOpts;
  IdentifierTable ToIdentifierTable;
  SelectorTable ToSelectorTable;
  ASTContext ToContext;
  FunctionDecl *ToFD;

  uint64_t NumInstances;
  uint64_t HashNum;
  bool CreatesTasks = false;
  bool UsesOmpif = false;
  bool MemcpyWideport = false;
  bool UsesLock = false;
  bool NeedsDeps = false;
  WrapperPortMap WrapperPortMap;

  llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo> Localmems;

  std::optional<uint64_t> getNumInstances() {
    uint64_t value = 1; // Default is 1 instance
    if (auto *numInstances =
            OriginalFD->getAttr<OSSTaskDeclAttr>()->getNumInstances()) {
      if (auto number = numInstances->getIntegerConstantExpr(OriginalContext);
          number && *number > 0) {
        value = number->getZExtValue();
      } else {
        Diag(numInstances->getExprLoc(),
             diag::err_expected_constant_unsigned_integer);
        return std::nullopt;
      }
    }

    return value;
  }

  std::optional<uint64_t> GenOnto() {
    auto *taskAttr = OriginalFD->getAttr<OSSTaskDeclAttr>();
    auto *ontoExpr = taskAttr->getOnto();
    // Check onto information
    if (ontoExpr) {
      auto ontoRes = ontoExpr->getIntegerConstantExpr(OriginalContext);
      if (!ontoRes || *ontoRes < 0) {
        Diag(ontoExpr->getExprLoc(),
             diag::err_expected_constant_unsigned_integer);
        return std::nullopt;
      }
      uint64_t onto = ontoRes->getZExtValue();
      return onto;
    }

    return clang::GenOnto(OriginalContext, OriginalFD);
  }

  void generateMemcpyWideportFunction(bool in) {

    const std::string memPtrType =
        "ap_uint<" +
        std::to_string(CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth) + ">";
    const std::string sizeofMemPtrType = "sizeof(" + memPtrType + ")";
    const std::string nElemsRead = "(sizeof(" + memPtrType + ")/sizeof(T))";

    Output << "template<class T>\n";
    Output
        << "void nanos6_fpga_memcpy_wideport_" << (in ? "in" : "out") << "(T * "
        << (in ? "dst" : "src")
        << ", const unsigned long long int addr, const unsigned int num_elems, "
        << memPtrType << "* mcxx_memport) {\n";
    Output << "#pragma HLS inline\n";
    Output << "  for (int i = 0; i < (num_elems-1)/" << nElemsRead
           << "+1; ++i) {\n";
    Output << "  #pragma HLS pipeline II=1\n";
    Output << "    " << memPtrType << " tmpBuffer;\n";
    if (in)
      Output << "    tmpBuffer = *(mcxx_memport + addr/" << sizeofMemPtrType
             << " + i);\n";
    Output << "    for (int j = 0; j < " << nElemsRead << "; ++j) {\n";
    if (CI.getFrontendOpts().OmpSsFpgaCheckLimitsMemoryPort) {
      Output << "      if (i*" << nElemsRead << "+j >= num_elems) break;\n";
    }
    Output << "      __mcxx_cast<T> cast_tmp;\n";
    if (in) {
      Output << "      cast_tmp.raw = tmpBuffer((j+1)*sizeof(T)*8-1, "
                "j*sizeof(T)*8);\n";
      Output << "      dst[i*" << nElemsRead << "+j] = cast_tmp.typed;\n";
    } else {
      Output << "      cast_tmp.typed = src[i*" << nElemsRead << "+j];\n";
      Output << "      tmpBuffer((j+1)*sizeof(T)*8-1, j*sizeof(T)*8) = "
                "cast_tmp.raw;\n";
    }
    Output << "    }\n";
    if (!in) {
      if (CI.getFrontendOpts().OmpSsFpgaCheckLimitsMemoryPort) {
        Output << "    const int rem = num_elems-(i*" << nElemsRead << ");\n";
        Output << "    const unsigned int bit_l = 0;\n";
        Output << "    const unsigned int bit_h = rem >= " << nElemsRead
               << " ? (" << sizeofMemPtrType
               << "*8-1) : ((rem*sizeof(T))*8-1);\n";
        Output << "    mcxx_memport[addr/sizeof(" << memPtrType << ") + i]"
               << "(bit_h, bit_l) = tmpBuffer(bit_h, bit_l);\n";
      } else {
        Output << "    *(mcxx_memport + addr/" << sizeofMemPtrType
               << " + i) = tmpBuffer;\n";
      }
    }
    Output << "  }\n}\n";
  }

  void GenerateWrapperHeader() {
    OutputHeaders << R"(///////////////////
// Automatic IP Generated by OmpSs@FPGA compiler
///////////////////
// The below code is composed by:
//  1) User source code, which may be under any license (see in original source code)
//  2) OmpSs@FPGA toolchain code which is licensed under LGPLv3 terms and conditions
///////////////////
)";
    OutputHeaders << "// Top IP Function: " << TaskFuncName << '\n';
    OutputHeaders << "// Accel. type hash: " << HashNum << '\n';
    OutputHeaders << "// Num. instances: " << NumInstances << '\n';
    OutputHeaders << "// Wrapper version: " << WrapperVersion << '\n';
    OutputHeaders << "///////////////////" << '\n';
  }

  void GenerateWrapperTop() {
    OutputHeaders << R"#(
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
)#";

    Output << "static ap_uint<64> " STR_TASKID ";\n";
    Output << R"(
template<class T>
union __mcxx_cast {
  unsigned long long int raw;
  T typed;
};
)";
    Output << R"(
struct mcxx_inaxis {
  ap_uint<64> data;
};
)";
    Output << R"(typedef ap_axiu<64, 1, 1, 2> mcxx_outaxis;
)";
    if (CreatesTasks) {
      Output << R"(
struct __fpga_copyinfo_t {
  unsigned long long int copy_address;
  unsigned char arg_idx;
  unsigned char flags;
  unsigned int size;
};
)";
      Output
          << "void mcxx_task_create(const ap_uint<64> type, const ap_uint<8> "
             "instanceNum, "
             "const ap_uint<8> numArgs, const unsigned long long int args[], "
             "const ap_uint<8> numDeps, const unsigned long long int deps[], "
             "const ap_uint<8> numCopies, const __fpga_copyinfo_t "
             "copies[], " STR_OUTPORT_DECL ");\n";

      Output << "void mcxx_taskwait(" STR_SPWNINPORT_DECL ", " STR_OUTPORT_DECL
                ");\n";
      Output << R"(
template <typename T>
struct __mcxx_ptr_t {
  unsigned long long int val;
  __mcxx_ptr_t(unsigned long long int val) : val(val) {}
  __mcxx_ptr_t() {}
  inline operator __mcxx_ptr_t<const T>() const {
    return __mcxx_ptr_t<const T>(val);
  }
  template <typename V> inline __mcxx_ptr_t<T> operator+(V const val) const {
    return __mcxx_ptr_t<T>(this->val + val*sizeof(T));
  }
  template <typename V> inline __mcxx_ptr_t<T> operator-(V const val) const {
    return __mcxx_ptr_t<T>(this->val - val*sizeof(T));
  }
  template <typename V> inline operator V() const {
    return (V)val;
  }
};
)";
    }

    if (UsesLock) {
      Output << "void mcxx_set_lock(" STR_INOUTPORT_DECL ");\n";
      Output << "void mcxx_unset_lock(" STR_OUTPORT_DECL ");\n";
    }

    if (WrapperPortMap[ToFD][size_t(WrapperPort::MEMORY_PORT)]) {
      generateMemcpyWideportFunction(true);
      generateMemcpyWideportFunction(false);
    }

    if (UsesOmpif) {
      Output << R"(
typedef enum {
  OMPIF_INT = 0,
  OMPIF_DOUBLE = 1,
  OMPIF_FLOAT = 2
} OMPIF_Datatype;
)";

      Output << R"(
typedef enum {
OMPIF_COMM_WORLD
} OMPIF_Comm;
)";

      Output << "void OMPIF_Send(const void *data, int count, OMPIF_Datatype "
                "datatype, int destination, unsigned char tag, OMPIF_Comm "
                "communicator, const ap_uint<8> numDeps, const unsigned long "
                "long int deps[], " STR_OUTPORT_DECL ");\n";
      Output
          << "void OMPIF_Recv(void *data, int count, OMPIF_Datatype datatype,"
             " int source, unsigned char tag, OMPIF_Comm communicator, const "
             "ap_uint<8> numDeps, const unsigned long long int "
             "deps[], " STR_OUTPORT_DECL ");\n";
      Output << "void OMPIF_Allgather(void* data, int count, OMPIF_Datatype "
                "datatype, unsigned char tag, OMPIF_Comm communicator, unsigned"
                "char ompif_rank, " STR_SPWNINOUTPORT_DECL ");\n";
    }
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      Output << "typedef ap_uint<105> __mcxx_instrData_t;\n"
             << "void mcxx_instrument_event(unsigned char event, unsigned long "
                "long payload, " STR_INSTRPORT_DECL ");\n";
    }
  }

  bool GenOriginalFunctionMoved() {
    // Headers
    // We are only going to preserve the headers ending with .fpga.h or .fpga,
    // this was a restriction in the original ompss@fpga and it simplifies some
    // tasks in a major way.
    for (auto &&file : PP.getIncludedFiles()) {
      if (!file->getName().endswith(".fpga.h") &&
          !file->getName().endswith(".fpga")) {
        continue;
      }
      StringRef headerInclude;
      // We need to generate a string with two " and the path
      std::string headerIncludeStorage =
          std::string("\"").append(file->tryGetRealPathName()).append("\"");
      headerInclude = headerIncludeStorage;
      if (headerInclude.size() == 2) /*Missing path*/ {
        // This extracts the path from the source file
        auto loc = OriginalSourceMgr.getIncludeLoc(
            OriginalSourceMgr.translateFile(file));
        auto *data = OriginalSourceMgr.getCharacterData(loc);
        // Just to make sure I don't overrun a buffer later
        auto bufferData =
            OriginalSourceMgr.getBufferData(OriginalSourceMgr.getFileID(loc));
        const auto *endChar = bufferData.data() + bufferData.size();
        auto *dataEndInclude = data;
        for (; dataEndInclude < endChar && *dataEndInclude != '\n' &&
               *dataEndInclude != '\r' && *dataEndInclude != '\0';
             ++dataEndInclude)
          ;

        headerInclude = StringRef(data, dataEndInclude - data);
      }
      OutputHeaders << "#include " << headerInclude << '\n';
    }

    // Body functions
    ToFD->dropAttr<OSSTaskDeclAttr>();
    auto isProtected = [&](StringRef name) {
      return name == "OMPIF_Comm_rank" || name == "OMPIF_Comm_size" ||
             name == "OMPIF_Send" || name == "OMPIF_Recv" ||
             name == "OMPIF_Allgather" ||
             name.starts_with("nanos6_fpga_memcpy_wideport_");
    };
    for (Decl *otherDecl : ToContext.getTranslationUnitDecl()->decls()) {
      if (ToSourceManager.getFileID(otherDecl->getSourceRange().getBegin()) !=
          ToSourceManager.getFileID(ToFD->getSourceRange().getBegin())) {
        // Skip dependency not originating in the file.
        continue;
      }
      if (auto *funcDecl = dyn_cast<FunctionDecl>(otherDecl);
          funcDecl && !funcDecl->hasBody() &&
          !isProtected(funcDecl->getName())) {
        Diag(otherDecl->getLocation(),
             diag::err_oss_fpga_missing_body_for_function_depended_by_kernel);
        Diag(ToFD->getLocation(), diag::note_oss_fpga_kernel);
        return false;
      } else if (funcDecl && !isProtected(funcDecl->getName()) &&
                 !funcDecl->hasAttr<OSSTaskDeclAttr>()) {
        auto origName = funcDecl->getDeclName();
        auto &id =
            PP.getIdentifierTable().get(origName.getAsString() + "_moved");

        DeclarationName name(&id);
        funcDecl->setDeclName(name);
      }
    }

    auto [needsDeps, replacementMap] = OmpssFpgaTreeTransform(
        ToContext, ToIdentifierTable, WrapperPortMap,
        CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth, CreatesTasks,
        CI.getFrontendOpts().OmpSsFpgaInstrumentation);
    NeedsDeps = needsDeps;
    ReplacementMap = std::move(replacementMap);
    for (Decl *otherDecl : ToContext.getTranslationUnitDecl()->decls()) {
      if (ToSourceManager.getFileID(otherDecl->getSourceRange().getBegin()) !=
          ToSourceManager.getFileID(ToFD->getSourceRange().getBegin())) {
        // Skip dependency not originating in the file.
        continue;
      }

      if (llvm::isa<FunctionDecl>(otherDecl)) {
        auto *FuncDecl = cast<FunctionDecl>(otherDecl);
        if (FuncDecl->hasAttr<OSSTaskDeclAttr>()) { // Don't print tasks, as now
                                                    // they are not needed in
                                                    // the wrapper
          continue;
        }
      }
      otherDecl->print(Output, printPol, 0, true);
      if (!OutputStr.empty() && OutputStr[OutputStr.size() - 1] != '\n') {
        Output << ";\n"; // This may lead to superfluous ';', but I'm not
                         // quite sure how to prevent them easily.
                         // Trailing ; are legal fortunately!
      }
    }
    return true;
  }

  std::string GetDeclVariableString(StorageClass storageClass, QualType type,
                                    StringRef name) {
    auto &id = PP.getIdentifierTable().get(name);

    auto *varDecl = VarDecl::Create(ToContext, ToFD->getLexicalDeclContext(),
                                    SourceLocation(), SourceLocation(), &id,
                                    type, nullptr, storageClass);

    std::string res;
    llvm::raw_string_ostream stream(res);
    varDecl->print(stream, printPol);
    return res;
  }

  void GenerateWrapperFunctionLocalmems() {
    for (auto &&p : Localmems) {
      Output << "  "
             << GetDeclVariableString(
                    StorageClass::SC_Static,
                    LocalmemArrayType(OriginalContext, p.second.FixedArrayRef),
                    p.first->getName())
             << ";\n";
    }
  }

  void GenerateWrapperFunctionParams() {
    Output << "void " << OrigFuncName << "_wrapper(" STR_INOUTPORT_DECL;

    if (CreatesTasks) {
      Output << ", " STR_SPWNINPORT_DECL;
    }

    bool forceMemport = [&] {
      auto *it = WrapperPortMap.find(ToFD);
      return it != WrapperPortMap.end() &&
             it->second[(int)WrapperPort::MEMORY_PORT];
    }();

    if (!CreatesTasks) {
      if ((CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth > 0 &&
           Localmems.size() > 0) ||
          forceMemport) {
        Output << ", ap_uint<" << CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth
               << ">* mcxx_memport";
      }

      for (auto *param : OriginalFD->parameters()) {
        auto it = Localmems.find(param);
        if (CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth > 0 &&
            it != Localmems.end())
          continue;

        QualType paramType = param->getType();
        if (paramType->isPointerType() || paramType->isArrayType()) {
          Output << ", "
                 << GetDeclVariableString(
                        StorageClass::SC_None,
                        OriginalContext.getPointerType(
                            GetElementTypePointerTo(paramType)),
                        " mcxx_" + param->getNameAsString());
        }
      }
    }
    if (UsesOmpif) {
      Output << ", unsigned char ompif_rank, unsigned char ompif_size";
    }
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      Output << ", " STR_INSTRPORT_DECL;
    }
    Output << ") {\n";

    Output << "#pragma HLS interface ap_ctrl_none port=return\n";
    Output << "#pragma HLS interface axis port=" STR_INPORT "\n";
    Output << "#pragma HLS interface axis port=" STR_OUTPORT "\n";
    if (CreatesTasks) {
      Output << "#pragma HLS interface axis port=" STR_SPWNINPORT "\n";
    }
    if (UsesOmpif) {
      Output << "#pragma HLS interface ap_stable port=ompif_rank\n";
      Output << "#pragma HLS interface ap_stable port=ompif_size\n";
    }

    if (!CreatesTasks) {
      if ((CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth > 0 &&
           Localmems.size() > 0) ||
          forceMemport) {
        Output << "#pragma HLS interface m_axi port=mcxx_memport\n";
      }
      for (auto *param : OriginalFD->parameters()) {
        auto it = Localmems.find(param);
        if (CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth > 0 &&
            it != Localmems.end())
          continue;

        QualType paramType = param->getType();
        if (paramType->isPointerType() || paramType->isArrayType()) {
          Output << "#pragma HLS interface m_axi port=mcxx_" << param->getName()
                 << "\n";
        }
      }
    }
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      Output << "#pragma HLS interface ap_hs port=" STR_INSTRPORT "\n";
    }
  }

  void GenerateWrapperFunctionParamReads() {

    auto getType = [&](ParmVarDecl *param) -> std::pair<QualType, bool> {
      auto paramType = param->getType();
      bool usesMemoryPort = false;
      if (paramType->isPointerType() || paramType->isArrayType()) {
        usesMemoryPort = true;
      }
      if (paramType->isArrayType()) {
        paramType =
            OriginalContext.getPointerType(GetElementTypePointerTo(paramType));
      }
      if (!usesMemoryPort && paramType.isConstQualified()) {
        paramType.removeLocalConst();
      }
      return std::pair{paramType, usesMemoryPort};
    };
    auto paramId = 0;
    for (auto *param : OriginalFD->parameters()) {
      auto [paramType, _] = getType(param);
      StringRef symbolName = param->getName();
      if (CreatesTasks && paramType->isPointerType()) {
        Output << "  __mcxx_ptr_t<";
        paramType->getPointeeType().print(Output, printPol);
        Output << "> " << symbolName << ";\n";
      } else {
        auto it = Localmems.find(param);
        if (it == Localmems.end()) {
          if (paramType->isArrayType()) {
            paramType = OriginalContext.getPointerType(
                GetElementTypePointerTo(paramType));
          }
          Output << "  "
                 << GetDeclVariableString(StorageClass::SC_None, paramType,
                                          symbolName)
                 << ";\n";
        } else {
          Output << "  ap_uint<8> mcxx_flags_" << paramId << ";\n";
          Output << "  ap_uint<64> mcxx_offset_" << paramId << ";\n";
        }
      }
      ++paramId;
    }
    Output << "  {\n";
    Output << "  #pragma HLS protocol fixed\n";
    paramId = 0;
    for (auto *param : OriginalFD->parameters()) {
      auto [paramType, usesMemoryPort] = getType(param);

      auto symbolName = param->getName();
      auto it = Localmems.find(param);
      Output << "    {\n";
      if (!usesMemoryPort || (CreatesTasks && paramType->isPointerType()) ||
          it == Localmems.end()) {
        Output << "      ap_uint<8> mcxx_flags_" << paramId << ";\n";
        Output << "      ap_uint<64> mcxx_offset_" << paramId << ";\n";
      }
      Output << "      mcxx_flags_" << paramId << " = "
             << STR_INPORT_READ "(7,0);\n";
      Output << "      ap_wait();\n";
      if (usesMemoryPort) {
        Output << "      mcxx_offset_" << paramId
               << " = " STR_INPORT_READ ";\n";
        if (CreatesTasks && paramType->isPointerType()) {
          Output << "      " << symbolName << ".val = mcxx_offset_" << paramId
                 << ";\n";
        } else if (it == Localmems.end()) {
          QualType pointedType = paramType->getPointeeType();
          Output << "      " << symbolName << " = mcxx_" << symbolName
                 << " + mcxx_offset_" << paramId << "/sizeof(";
          pointedType.print(Output, printPol);
          Output << ");\n";
        } else {
          it->second.ParamIdx = paramId;
        }
      } else {
        Output << "      __mcxx_cast<";
        paramType.print(Output, printPol);
        Output << "> mcxx_arg_" << paramId << ";\n";

        Output << "      mcxx_arg_" << paramId
               << ".raw = " STR_INPORT_READ ";\n";
        Output << "      " << symbolName << " = mcxx_arg_" << paramId
               << ".typed;\n";
      }
      Output << "    }\n";
      Output << "    ap_wait();\n";
      ++paramId;
    }
    Output << "  }\n";
  }

  void GenerateWrapperFunctionLocalmemCopies(LocalmemInfo::Dir dir) {
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      if (dir & LocalmemInfo::Dir::IN) {
        Output << "mcxx_instrument_event("
               << uint32_t(FPGAInstrumentationEvents::DevCopyInBegin)
               << ", " STR_TASKID ", " STR_INSTRPORT ");\n";
      } else {
        Output << "mcxx_instrument_event("
               << uint32_t(FPGAInstrumentationEvents::DevCopyOutBegin)
               << ", " STR_TASKID ", " STR_INSTRPORT ");\n";
      }
    }

    for (auto &&[param, localmemInfo] : Localmems) {
      if ((localmemInfo.dir & dir) == 0) {
        continue;
      }

      const auto &fixedArrayRef = localmemInfo.FixedArrayRef;
      int paramId = localmemInfo.ParamIdx;
      QualType baseType = GetElementTypePointerTo(param->getType());
      uint64_t baseTypeSize = OriginalContext.getTypeSize(baseType) /
                              OriginalContext.getCharWidth();

      const auto paramName = param->getName();
      const auto baseTypeSizeStr = std::to_string(baseTypeSize);
      const std::string memPtrType =
          "ap_uint<" +
          (CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth > 0
               ? std::to_string(CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth)
               : std::to_string(baseTypeSize * 8)) +
          ">";
      const std::string sizeofMemPtrType = "sizeof(" + memPtrType + ")";

      const std::string dataReferenceSize =
          "(" +
          std::to_string(ComputeArrayRefSize(OriginalContext, fixedArrayRef,
                                             baseTypeSize)) +
          ")";
      const std::string nElementsSrc =
          "(" + dataReferenceSize + "/" + baseTypeSizeStr + ")";
      const std::string nElemsRead =
          "(sizeof(" + memPtrType + ")/" + baseTypeSizeStr + ")";

      Output << "  if (mcxx_flags_" << paramId << "["
             << (dir == LocalmemInfo::IN ? STR_ARG_FLAG_IN_BIT
                                         : STR_ARG_FLAG_OUT_BIT)
             << "]) {\n";
      if (CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth == 0) {
        QualType pointedType = GetElementTypePointerTo(param->getType());
        if (dir == LocalmemInfo::IN) {
          Output << "    memcpy(" << paramName << ", mcxx_" << paramName
                 << " + mcxx_offset_" << paramId << "/sizeof(";
          pointedType.print(Output, printPol);
          Output << "), " << dataReferenceSize << ");\n";
        } else {
          Output << "    memcpy(mcxx_" << paramName << " + mcxx_offset_"
                 << paramId << "/sizeof(";
          pointedType.print(Output, printPol);
          Output << "), " << paramName << ", " << dataReferenceSize << ");\n";
        }
      } else {
        Output << "    for (int __i = 0; "
               << "__i < (" << dataReferenceSize << " - 1)/" << sizeofMemPtrType
               << "+1; "
               << "++__i) {\n";
        Output << "    #pragma HLS pipeline II=1\n";
        Output << "      " << memPtrType << " __tmpBuffer;\n";
        if (dir == LocalmemInfo::IN) {
          Output << "      __tmpBuffer = *(mcxx_memport + mcxx_offset_"
                 << paramId << '/' << sizeofMemPtrType << " + __i);\n";
        }
        Output << "      for (int __j=0; "
               << "__j <" << nElemsRead << "; "
               << "__j++) {\n";
        if (CI.getFrontendOpts().OmpSsFpgaCheckLimitsMemoryPort) {
          Output << "        if (__i*" << nElemsRead
                 << "+__j >= " << nElementsSrc << ") continue;\n";
        }
        Output << "        __mcxx_cast<";
        baseType.print(Output, printPol);
        Output << "> cast_tmp;\n";
        if (dir == LocalmemInfo::IN) {
          Output << "        cast_tmp.raw = __tmpBuffer("
                 << "(__j+1)*" << baseTypeSizeStr << "*8-1,"
                 << "__j*" << baseTypeSizeStr << "*8);\n";
          Output << "        " << paramName << "[__i*" << nElemsRead
                 << "+__j] = cast_tmp.typed;\n";
        } else {
          Output << "        "
                 << "cast_tmp.typed = " << paramName << "[__i*" << nElemsRead
                 << "+__j];\n";
          Output << "        "
                 << "__tmpBuffer((__j+1)*" << baseTypeSizeStr << "*8-1,"
                 << "__j*" << baseTypeSizeStr << "*8) = cast_tmp.raw;\n";
        }
        Output << "      }\n";
        if (dir == LocalmemInfo::OUT) {
          if (CI.getFrontendOpts().OmpSsFpgaCheckLimitsMemoryPort) {
            Output << "      "
                   << "const int rem = " << nElementsSrc << "-(__i*"
                   << nElemsRead << ");\n";
            Output << "      "
                   << "const unsigned int bit_l = 0;\n";
            Output << "      "
                   << "const unsigned int bit_h = rem >= " << nElemsRead
                   << " ? (" << sizeofMemPtrType << "*8-1) : ((rem*"
                   << baseTypeSizeStr << ")*8-1);\n";
            Output << "      "
                   << "mcxx_memport[mcxx_offset_" << paramId << '/'
                   << sizeofMemPtrType << "+ __i]"
                   << "(bit_h, bit_l) = __tmpBuffer(bit_h, bit_l);\n";
          } else {
            Output << "      "
                   << "*(mcxx_memport + mcxx_offset_" << paramId << '/'
                   << sizeofMemPtrType << "+ __i) = __tmpBuffer;\n";
          }
        }
        Output << "    }\n";
      }
      Output << "  }\n";
    }
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      if (dir & LocalmemInfo::Dir::IN) {
        Output << "mcxx_instrument_event("
               << uint32_t(FPGAInstrumentationEvents::DevCopyInEnd)
               << ", " STR_TASKID ", " STR_INSTRPORT ");\n";
      } else {
        Output << "mcxx_instrument_event("
               << uint32_t(FPGAInstrumentationEvents::DevCopyOutEnd)
               << ", " STR_TASKID ", " STR_INSTRPORT ");\n";
      }
    }
  }

  void GenerateWrapperFunctionUserTaskCall() {
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      Output << "mcxx_instrument_event("
             << uint32_t(FPGAInstrumentationEvents::DevExecBegin)
             << ", " STR_TASKID ", " STR_INSTRPORT ");\n";
    }
    auto &outs = Output << "  " << TaskFuncName << "(";
    bool first = true;
    auto printSeparator = [&]() {
      if (first) {
        first = false;
      } else {
        outs << ", ";
      }
    };

    for (auto *param : OriginalFD->parameters()) {
      printSeparator();
      outs << param->getName();
    }
    auto *it = WrapperPortMap.find(ToFD);
    if (it != WrapperPortMap.end()) {
      if (it->second[(int)WrapperPort::OMPIF_RANK]) {
        printSeparator();
        outs << "ompif_rank";
      }
      if (it->second[(int)WrapperPort::OMPIF_SIZE]) {
        printSeparator();
        outs << "ompif_size";
      }
      if (it->second[(int)WrapperPort::SPAWN_INPORT]) {
        printSeparator();
        outs << STR_SPWNINPORT;
      }
      if (it->second[(int)WrapperPort::INPORT]) {
        printSeparator();
        outs << STR_INPORT;
      }
      if (it->second[(int)WrapperPort::OUTPORT]) {
        printSeparator();
        outs << STR_OUTPORT;
      }
      if (it->second[(int)WrapperPort::MEMORY_PORT]) {
        printSeparator();
        outs << "mcxx_memport";
      }
    }
    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      printSeparator();
      outs << STR_INSTRPORT;
    }
    outs << ");\n";

    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      outs << "mcxx_instrument_event("
           << uint32_t(FPGAInstrumentationEvents::DevExecEnd)
           << ", " STR_TASKID ", " STR_INSTRPORT ");\n";
    }
  }

  void GenerateWrapperFunction() {
    if (CI.getFrontendOpts().OmpSsFpgaMemoryPortWidth == 0 &&
        !Localmems.empty()) {
      OutputHeaders << "#include <string.h> //needed for memcpy\n";
    }
    Output << "void mcxx_write_out_port(const ap_uint<64> data, const "
              "ap_uint<2> dest, const ap_uint<1> last, " STR_OUTPORT_DECL ") {";
    Output << R"(
  #pragma HLS inline
  mcxx_outaxis axis_word;
  axis_word.data = data;
  axis_word.dest = dest;
  axis_word.last = last;
)";
    Output << "  " STR_OUTPORT_WRITE("axis_word") ";\n";
    Output << "}\n";

    GenerateWrapperFunctionParams();
    GenerateWrapperFunctionLocalmems();

    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      Output << "  ap_uint<64> __command = " STR_INPORT_READ
                "; //command word\n";
      Output << "  if (__command(7,0) == 2) {\n";
      Output << "    __mcxx_instrData_t tmpSetup;\n";
      Output << "    tmpSetup(63,0) = " STR_INPORT_READ ";\n";
      Output << "    tmpSetup(79,64) =  (__command>>8)&0xFFFFFF;\n";
      Output << "    tmpSetup[104] = 0;\n";
      Output << "    " STR_INSTRPORT ".write(tmpSetup);\n";
      Output << "    return;\n";
      Output << "  }\n";
    } else {
      Output << "  " STR_INPORT_READ_NODATA "; //command word\n";
    }
    Output << "  " STR_TASKID " = " STR_INPORT_READ ";\n";
    Output << "  ap_uint<64> " STR_PARENT_TASKID " = " STR_INPORT_READ ";\n";

    GenerateWrapperFunctionParamReads();
    GenerateWrapperFunctionLocalmemCopies(LocalmemInfo::IN);
    GenerateWrapperFunctionUserTaskCall();
    GenerateWrapperFunctionLocalmemCopies(LocalmemInfo::OUT);
    Output << "  {\n"; // send finish task
    Output << "  #pragma HLS protocol fixed\n";
    Output << "    ap_uint<64> header = 0x03;\n";
    Output << "    ap_wait();\n";
    Output << "    " STR_OUTPORT_WRITE_FUN("header, " STR_FINISH_TASK_CODE
                                           ", 0")
           << "\n";
    Output << "    "
              "ap_wait();\n";
    Output << "    " STR_OUTPORT_WRITE_FUN(STR_TASKID ", " STR_FINISH_TASK_CODE
                                                      ", 0")
           << "\n";

    Output << "    "
              "ap_wait();\n";
    Output << "    " STR_OUTPORT_WRITE_FUN(STR_PARENT_TASKID
                                           ", " STR_FINISH_TASK_CODE ", 1")
           << "\n";
    Output << "    "
              "ap_wait();\n";
    Output << "  }\n";
    Output << "}\n";
  }

  void GenerateWrapperBottom() {
    if (CreatesTasks) {
      Output
          << "void mcxx_task_create(const ap_uint<64> type, const ap_uint<8> "
             "instanceNum, "
             "const ap_uint<8> numArgs, const unsigned long long int args[], "
             "const ap_uint<8> numDeps, const unsigned long long int deps[], "
             "const ap_uint<8> numCopies, const __fpga_copyinfo_t "
             "copies[], " STR_OUTPORT_DECL ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  const ap_uint<2> destId = " STR_NEW_TASK_CODE ";\n";
      Output << "  ap_uint<64> tmp;\n";
      Output << "  tmp(15,8)  = numArgs;\n";
      Output << "  tmp(23,16) = numDeps;\n";
      Output << "  tmp(31,24) = numCopies;\n";
      Output << "  " STR_OUTPORT_WRITE_FUN("tmp, destId, 0") "\n";
      Output << "  " STR_OUTPORT_WRITE_FUN(STR_TASKID ", destId, 0") "\n";
      Output << "  tmp(47,40) = instanceNum;\n";
      Output << "  tmp(33,0)  = type(33,0);\n";
      Output << "  " STR_OUTPORT_WRITE_FUN("tmp, destId, 0") "\n";
      Output << "  for (ap_uint<4> i = 0; i < numDeps(3,0); ++i) {\n";
      Output << "    " STR_OUTPORT_WRITE_FUN(
          "deps[i], destId, numArgs == 0 && numCopies == 0 && i == "
          "numDeps-1") "\n";
      Output << "  }\n";
      Output << "  for (ap_uint<4> i = 0; i < numCopies(3,0); ++i) {\n";
      Output << "    " STR_OUTPORT_WRITE_FUN(
          "copies[i].copy_address, destId, 0") "\n";
      Output << "    tmp(7,0) = copies[i].flags;\n";
      Output << "    tmp(15,8) = copies[i].arg_idx;\n";
      Output << "    tmp(63,32) = copies[i].size;\n";
      Output << "    " STR_OUTPORT_WRITE_FUN(
          "tmp, destId, numArgs == 0 && i == numCopies-1") "\n";
      Output << "  }\n";
      Output << "  for (ap_uint<4> i = 0; i < numArgs(3,0); ++i) {\n";
      Output << "    " STR_OUTPORT_WRITE_FUN(
          "args[i], destId, i == numArgs-1") "\n";
      Output << "  }\n";
      Output << "}\n";
      Output << "\n"; // blank line
      Output << "void mcxx_taskwait(" STR_SPWNINPORT_DECL ", " STR_OUTPORT_DECL
                ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  ap_wait();\n";
      Output << "  " STR_OUTPORT_WRITE_FUN(STR_TASKID ", " STR_TASKWAIT_CODE
                                                      ", 1") "\n";
      Output << "  ap_wait();\n";
      Output << "  " STR_SPWNINPORT_READ ";\n";
      Output << "  ap_wait();\n";
      Output << "}\n";
    }
    if (UsesLock) {
      Output << "void mcxx_set_lock(" STR_INOUTPORT_DECL ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  ap_uint<64> tmp = 0x4;\n";
      Output << "  ap_uint<8> ack;\n";
      Output << "  do {\n";
      Output << "    ap_wait();\n";
      Output << "    " STR_OUTPORT_WRITE_FUN("tmp, 1, 1") "\n";
      Output << "    ap_wait();\n";
      Output << "    ack = " STR_INPORT_READ ";\n";
      Output << "    ap_wait();\n";
      Output << "  } while (ack == 0);\n";
      Output << "}\n";
      Output << "\n"; // blank line
      Output << "void mcxx_unset_lock(" STR_OUTPORT_DECL ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  ap_uint<64> tmp = 0x6;\n";
      Output << "  " STR_OUTPORT_WRITE_FUN("tmp, 1, 1") "\n";
      Output << "}\n";
    }
    if (UsesOmpif) {
      Output << "const int ompif_type_sizes[3] = {sizeof(int), sizeof(double), "
                "sizeof(float)};\n";
      Output << "void OMPIF_Send(const void *data, int count, OMPIF_Datatype "
                "datatype, int destination, unsigned char tag, OMPIF_Comm "
                "communicator, const ap_uint<8> numDeps, const unsigned long "
                "long int deps[], " STR_OUTPORT_DECL ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  ap_uint<64> command;\n";
      Output << "  command(7,0) = 0;\n";
      Output << "  command(15,8) = tag;\n";
      Output << "  command(23,16) = destination+1;\n";
      Output << "  command(63, 32) = (unsigned long long int)data;\n";
      Output
          << "  unsigned long long int args[2] = {command, (unsigned long long "
             "int)count*ompif_type_sizes[(int)datatype]};\n";
      Output << "  " STR_TASK_CREATE_FUN(
          "4294967299LU, 0xFF, 2, args, numDeps, deps, 0, 0") "\n";
      Output << "}\n";
      Output
          << "void OMPIF_Recv(void *data, int count, OMPIF_Datatype datatype, "
             "int source, unsigned char tag, OMPIF_Comm communicator, const "
             "ap_uint<8> numDeps, const unsigned long long int "
             "deps[], " STR_OUTPORT_DECL ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  ap_uint<64> command;\n";
      Output << "  command(7,0) = 0;\n";
      Output << "  command(15,8) = tag;\n";
      Output << "  command(23,16) = source+1;\n";
      Output << "  command(63, 32) = (unsigned long long int)data;\n";
      Output
          << "unsigned long long int args[2] = {command, (unsigned long long "
             "int)count*ompif_type_sizes[(int)datatype]};\n";
      Output << "  " STR_TASK_CREATE_FUN(
          "4294967300LU, 0xFF, 2, args, numDeps, deps, 0, 0") "\n";
      Output << "}\n";
      Output
          << "void OMPIF_Allgather(void *data, int count, OMPIF_Datatype "
             "datatype, unsigned char tag, OMPIF_Comm communicator, unsigned "
             "char ompif_rank, " STR_SPWNINOUTPORT_DECL ") {\n";
      Output << "#pragma HLS inline\n";
      Output << "  ap_uint<64> command_sender, command_receiver;\n";
      Output << "  const unsigned long long int size = (unsigned long long "
                "int)count*ompif_type_sizes[(int)datatype];\n";
      Output << "  command_sender(7,0) = 1; //SENDALL\n";
      Output << "  command_receiver(7, 0) = 1; //RECVALL\n";
      Output << "  command_sender(15, 8) = tag; //TAG\n";
      Output << "  command_receiver(15, 8) = tag;\n";
      Output << "  command_sender(63, 32) = (unsigned long long int)data + "
                "(ompif_rank-1)*size;\n";
      Output << "  command_receiver(63, 32) = (unsigned long long int)data;\n";
      Output << "  const unsigned long long int mcxx_args_sender[2] = "
                "{command_sender, size};\n";
      Output << "  const unsigned long long int mcxx_args_receiver[2] = "
                "{command_receiver, size};\n";
      Output << "  " STR_TASK_CREATE_FUN(
          "4294967300LU, 255, 2, mcxx_args_receiver, 0, NULL, 0, NULL") "\n";
      Output << "  " STR_TASK_CREATE_FUN(
          "4294967299LU, 255, 2, mcxx_args_sender, 0, NULL, 0, NULL") "\n";
      Output << "  " STR_TASKWAIT_FUN "\n";
      Output << "}\n";
    }

    if (CI.getFrontendOpts().OmpSsFpgaInstrumentation) {
      Output << "void mcxx_instrument_event(unsigned char event, unsigned long "
                "long payload, " STR_INSTRPORT_DECL ") {\n"
                "#pragma HLS inline\n"
                "  __mcxx_instrData_t tmp;\n"
                "  tmp.range(63, 0) = payload;\n"
                "  tmp.range(95, 64) = event & 0x7F;\n"
                "  tmp.range(103, 96) = event >> 7;\n"
                "  tmp.bit(104) = 1;\n"
                "  ap_wait();\n"
                "  " STR_INSTRPORT ".write(tmp);\n"
                "  ap_wait();\n"
                "}\n";
    }
  }

  bool generateMovedContext() {
    // Dependency resolution. We use the ASTImporter utility, which is able to
    // manage any sort of C++ construct during resolution.
    ASTImporter importer(ToContext, ToFileManager, OriginalContext,
                         OriginalSourceMgr.getFileManager(), true);

    auto importedOrErr = importer.Import(OriginalFD);
    if (!importedOrErr) {
      auto err = importedOrErr.takeError();
      std::string out;
      llvm::raw_string_ostream stream(out);
      stream << err;
      Diag(OriginalFD->getLocation(), diag::err_oss_fpga_dependency_analisis)
          << out;
      return false;
    }
    ToFD = dyn_cast<FunctionDecl>(*importedOrErr);
    return true;
  }

public:
  WrapperGenerator(Callable Diag, llvm::raw_ostream &OutputFile,
                   FunctionDecl *FD, llvm::StringRef FuncName,
                   ASTContext &SourceContext, SourceManager &SourceMgr,
                   Preprocessor &PP, CompilerInstance &CI)
      : Diag(std::forward<Callable>(Diag)), OutputHeaders(OutputStrHeaders),
        Output(OutputStr), OutputFinalFile(OutputFile), OrigFuncName(FuncName),
        TaskFuncName(std::string(FuncName) + "_moved"), PP(PP), CI(CI),
        printPol(SourceContext.getLangOpts()), OriginalFD(FD),
        OriginalContext(SourceContext), OriginalSourceMgr(SourceMgr),
        ToDiagnosticsEngine(new DiagnosticIDs(), new DiagnosticOptions()),
        ToFileManager(SourceMgr.getFileManager().getFileSystemOpts()),
        ToSourceManager(ToDiagnosticsEngine, ToFileManager),
        ToLangOpts(SourceContext.getLangOpts()), ToIdentifierTable(),
        ToSelectorTable(),
        ToContext(ToLangOpts, ToSourceManager, ToIdentifierTable,
                  ToSelectorTable, PP.getBuiltinInfo(), TU_Incremental) {
    printPol.adjustForCPlusPlus();
    printPol.Replacements = &ReplacementMap;
    ToContext.InitBuiltinTypes(OriginalContext.getTargetInfo());
  }

  bool GenerateWrapperFile() {
    auto numInstances = getNumInstances();
    if (!numInstances) {
      return false;
    }
    NumInstances = std::move(*numInstances);

    Localmems = ComputeLocalmems(OriginalFD);

    auto hashNum = GenOnto();
    if (!hashNum) {
      return false;
    }
    HashNum = std::move(*hashNum);

    if (!generateMovedContext()) {
      return false;
    }

    FPGAFunctionTreeVisitor visitor(ToFD, WrapperPortMap);
    visitor.TraverseStmt(ToFD->getBody());

    MemcpyWideport = visitor.MemcpyWideport;
    CreatesTasks = visitor.CreatesTasks;
    UsesLock = visitor.UsesLock;
    UsesOmpif = visitor.UsesOmpif;

    GenerateWrapperHeader();
    GenerateWrapperTop();
    if (!GenOriginalFunctionMoved()) {
      return false;
    }
    GenerateWrapperFunction();
    GenerateWrapperBottom();
    OutputFinalFile << OutputStrHeaders;
    OutputFinalFile << OutputStr;
    return true;
  }

  void AddPartJson(llvm::raw_ostream &OutputJson, StringRef generatedPathFile,
                   StringRef generatedFile) {
    OutputJson << "{\n";
    OutputJson << "    \"full_path\" : \"" << generatedPathFile << "\",\n";
    OutputJson << "    \"filename\" : \"" << generatedFile << "\",\n";
    OutputJson << "    \"name\" : \"" << OrigFuncName << "\",\n";
    OutputJson << "    \"type\" : " << HashNum << ",\n";
    OutputJson << "    \"num_instances\" : " << NumInstances << ",\n";
    OutputJson << "    \"task_creation\" : "
               << (CreatesTasks ? "true" : "false") << ",\n";
    OutputJson << "    \"instrumentation\" : "
               << (CI.getFrontendOpts().OmpSsFpgaInstrumentation ? "true"
                                                                 : "false")
               << ",\n";
    OutputJson << "    \"periodic\" : false,\n";
    OutputJson << "    \"lock\" : " << (UsesLock ? "true" : "false") << ",\n";
    OutputJson << "    \"deps\" : " << (NeedsDeps ? "true" : "false") << ",\n";
    OutputJson << "    \"ompif\" : " << (UsesOmpif ? "true" : "false") << "\n";
    OutputJson << "},\n";
  }
};
}
void FPGAWrapperGen::ActOnOmpSsFpgaExtractFiles(clang::ASTContext &Ctx) {
  if (Ctx.ompssFpgaDecls.empty())
    return;

  auto diag = [&](auto... arg) { return PP.Diag(arg...); };
  auto realPathOrNone = getAbsoluteDirExport(
      Ctx.getSourceManager(), CI.getFrontendOpts().OmpSsFpgaHlsTasksDir,
      Ctx.ompssFpgaDecls[0]->getLocation(), diag);
  if (!realPathOrNone) {
    return;
  }

  for (auto *FD : Ctx.ompssFpgaDecls) {
    auto funcName = FD->getName();
    if (CI.getFrontendOpts().OmpSsFpgaDump) {
      llvm::outs() << funcName << '\n';
      if (!GenerateExtractedOriginalFunction(diag, llvm::outs(), FD, funcName,
                                             Ctx, Ctx.getSourceManager(), PP)) {
        return;
      }
    } else {
      std::ofstream stream{*realPathOrNone + "/" +
                           (funcName.str() + "_hls_automatic_clang." +
                            (Ctx.getLangOpts().CPlusPlus ? "cpp" : "c"))};
      llvm::raw_os_ostream outputFile(stream);

      if (!GenerateExtractedOriginalFunction(diag, outputFile, FD, funcName,
                                             Ctx, Ctx.getSourceManager(), PP)) {
        return;
      }
    }
  }
}

void FPGAWrapperGen::ActOnOmpSsFpgaGenerateWrapperCodeFiles(
    clang::ASTContext &Ctx) {
  if (Ctx.ompssFpgaDecls.empty())
    return;

  auto diag = [&](auto... arg) { return PP.Diag(arg...); };
  auto realPathOrNone = getAbsoluteDirExport(
      Ctx.getSourceManager(), CI.getFrontendOpts().OmpSsFpgaHlsTasksDir,
      Ctx.ompssFpgaDecls[0]->getLocation(), diag);
  if (!realPathOrNone) {
    return;
  }

  std::ofstream outputJson{*realPathOrNone + "/extracted.json.part"};
  llvm::raw_os_ostream outputJsonFile(outputJson);
  for (auto *decl : Ctx.ompssFpgaDecls) {
    auto *FD = dyn_cast<FunctionDecl>(decl);

    auto funcName = FD->getName();
    auto fileName = (funcName.str() + "_hls_automatic_clang.cpp");
    auto filePath = *realPathOrNone + "/" + fileName;
    if (CI.getFrontendOpts().OmpSsFpgaDump) {
      llvm::outs() << funcName << '\n';
      WrapperGenerator<decltype(diag)> wrapperGen(
          diag, llvm::outs(), FD, funcName, Ctx, Ctx.getSourceManager(), PP,
          CI);
      if (!wrapperGen.GenerateWrapperFile()) {
        return;
      }
      wrapperGen.AddPartJson(llvm::outs(), filePath, fileName);
    } else {
      std::ofstream stream{filePath};
      llvm::raw_os_ostream outputFile(stream);

      WrapperGenerator<decltype(diag)> wrapperGen(
          diag, outputFile, FD, funcName, Ctx, Ctx.getSourceManager(), PP, CI);
      if (!wrapperGen.GenerateWrapperFile()) {
        return;
      }
      wrapperGen.AddPartJson(outputJsonFile, filePath, fileName);
    }
  }
}

void FPGAWrapperGen::HandleTranslationUnit(clang::ASTContext &Ctx) {
  if (CI.getDiagnostics().hasErrorOccurred())
    return;
  if (CI.getFrontendOpts().OmpSsFpgaExtract &&
      CI.getFrontendOpts().OmpSsFpgaWrapperCode) {
    CI.getDiagnostics().Report(diag::err_oss_fpga_wrapper_code_and_extract);
    return;
  }
  if (CI.getFrontendOpts().OmpSsFpgaExtract) {
    ActOnOmpSsFpgaExtractFiles(Ctx);
  }
  if (CI.getFrontendOpts().OmpSsFpgaWrapperCode) {
    ActOnOmpSsFpgaGenerateWrapperCodeFiles(Ctx);
  }
}

FPGAWrapperGen::FPGAWrapperGen(Preprocessor &PP, CompilerInstance &CI)
    : PP(PP), CI(CI) {}
FPGAWrapperGen::~FPGAWrapperGen() = default;