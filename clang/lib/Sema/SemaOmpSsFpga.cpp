//===--- SemaOmpSsFpga.cpp - Semantic Analysis for OmpSs Fpga target ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OmpSs FPGA functions and
/// generates other sources for following steps in the generation
///
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attrs.inc"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#define STR_COMPONENTS_COUNT "__mcxx_taskComponents"
#define STR_OUTPORT "mcxx_outPort"
#define STR_OUTPORT_WRITE(X) "mcxx_outPort.write(" X ")"
#define STR_INPORT "mcxx_inPort"
#define STR_SPWNINPORT "mcxx_spawnInPort"
#define STR_SPWNINPORT_READ "mcxx_spawnInPort.read()"
#define STR_INPORT_READ "mcxx_inPort.read()"
#define STR_INPORT_READ_NODATA "mcxx_inPort.read()"
#define STR_INPORT_TYPE "hls::stream<ap_uint<64> >"
#define STR_SPWNINPORT_TYPE "hls::stream<ap_uint<8> >"
#define STR_OUTPORT_TYPE "hls::stream<mcxx_outaxis>"
#define STR_INPORT_DECL STR_INPORT_TYPE "& " STR_INPORT
#define STR_SPWNINPORT_DECL STR_SPWNINPORT_TYPE "& " STR_SPWNINPORT
#define STR_OUTPORT_DECL STR_OUTPORT_TYPE "& " STR_OUTPORT
#define STR_INOUTPORT_DECL STR_INPORT_DECL ", " STR_OUTPORT_DECL
#define STR_SPWNINOUTPORT_DECL STR_SPWNINPORT_DECL ", " STR_OUTPORT_DECL
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
static constexpr auto WrapperVersion = 14;

enum class WrapperPort {
  OMPIF_RANK = 0,
  OMPIF_SIZE = 1,
  MEMORY_PORT = 2,
  INPORT = 3,
  OUTPORT = 4,
  SPAWN_INPORT = 5,
  NUM_PORTS = 6
};

using WrapperPortMap =
    llvm::SmallMapVector<const Decl *,
                         std::array<bool, size_t(WrapperPort::NUM_PORTS)>, 16>;

struct LocalmemInfo {
  int ParamIdx = -1;
  const OSSArrayShapingExpr *FixedArrayRef;
  enum Dir { IN = 0b01, OUT = 0b10, INOUT = 0b11 };
  Dir dir;
};

template <typename Callable>
Optional<std::string>
getAbsoluteDirExport(const SourceManager &SourceMgr, const std::string &path,
                     const SourceLocation &Location, Callable &&Diag) {
  auto &vfs = SourceMgr.getFileManager().getVirtualFileSystem();
  if (path.empty()) {
    Diag(Location, diag::err_oss_ompss_fpga_hls_tasks_dir_missing_dir);
    return llvm::NoneType{};
  }
  llvm::SmallVector<char, 128> realPathStr;
  if (vfs.getRealPath(path, realPathStr)) {
    Diag(Location, diag::err_oss_ompss_fpga_hls_tasks_dir_missing_dir);
    return llvm::NoneType{};
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

    llvm::StringRef headerInclude(data, dataEndInclude - data);
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

class FPGAFunctionTreeVisitor
    : public RecursiveASTVisitor<FPGAFunctionTreeVisitor> {
  struct FunctionCallTree {
    const Decl *symbol;
    FunctionCallTree *parent;

    FunctionCallTree(const Decl *symbol, FunctionCallTree *parent)
        : symbol(symbol), parent(parent) {}
  };

  FunctionCallTree Top;
  FunctionCallTree *Current;
  WrapperPortMap &WrapperPortMap;

  void propagatePort(WrapperPort port) {
    const FunctionCallTree *node = Current;
    do {
      WrapperPortMap[node->symbol][(int)port] = true;
      node = node->parent;
    } while (node != nullptr);
  }

public:
  bool CreatesTasks = false;
  bool UsesOmpif = false;
  bool MemcpyWideport = false;
  bool UsesLock = false;

  FPGAFunctionTreeVisitor(FunctionDecl *startSymbol,
                          ::WrapperPortMap &wrapperPortMap)
      : Top(startSymbol, nullptr), Current(&Top),
        WrapperPortMap(wrapperPortMap) {}

  bool VisitOSSTaskDirective(OSSTaskDirective *) {
    CreatesTasks = true;
    propagatePort(WrapperPort::OUTPORT);
    return true;
  }

  bool VisitOSSTaskwaitDirective(OSSTaskwaitDirective *) {
    CreatesTasks = true;
    propagatePort(WrapperPort::OUTPORT);
    propagatePort(WrapperPort::SPAWN_INPORT);
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *n) {
    auto *body = n->getConstructor()->getBody();
    getDerived().VisitStmt(body);
    return true;
  }

  bool VisitOMPCriticalDirective(OMPCriticalDirective *n) {
    for (auto *c : n->children())
      getDerived().VisitStmt(c);

    UsesLock = true;
    propagatePort(WrapperPort::INPORT);
    propagatePort(WrapperPort::OUTPORT);
    return true;
  }

  bool VisitCallExpr(CallExpr *n) {
    for (auto *arg : n->children()) {
      getDerived().VisitStmt(arg);
    }
    auto *sym = n->getCalleeDecl();
    if (const NamedDecl *symNamed = dyn_cast<NamedDecl>(sym); symNamed) {
      auto symName = symNamed->getName();

      if (symName == "OMPIF_Comm_rank") {
        UsesOmpif = true;
        propagatePort(WrapperPort::OMPIF_RANK);
        return true;
      } else if (symName == "OMPIF_Comm_size") {
        UsesOmpif = true;
        propagatePort(WrapperPort::OMPIF_SIZE);
        return true;
      } else if (symName.startswith("OMPIF_")) {
        UsesOmpif = CreatesTasks = true;
        propagatePort(WrapperPort::OUTPORT);
        if (symName == "OMPIF_Allgather") {
          propagatePort(WrapperPort::OMPIF_RANK);
          propagatePort(WrapperPort::SPAWN_INPORT);
        }
      } else if (symName == "nanos6_fpga_memcpy_wideport_in" ||
                 symName == "nanos6_fpga_memcpy_wideport_out") {
        MemcpyWideport = true;
        propagatePort(WrapperPort::MEMORY_PORT);
        return true;
      }
    }
    auto *body = sym->getBody();
    if (!body) {
      return true;
    }

    FunctionCallTree newNode(sym, Current);
    FunctionCallTree *prev = Current;
    Current = &newNode;
    getDerived().VisitStmt(body);
    Current = prev;
    return true;
  }
};

template <typename Callable> class WrapperGenerator {
  // Construction
  Sema &SemaRef;
  Callable Diag;
  std::string OutputStrHeaders;
  llvm::raw_string_ostream OutputHeaders;
  std::string OutputStr;
  llvm::raw_string_ostream Output;
  llvm::raw_ostream &OutputFinalFile;
  FunctionDecl *FD;
  llvm::StringRef OrigFuncName;
  std::string TaskFuncName;
  ASTContext &SourceContext;
  SourceManager &SourceMgr;
  Preprocessor &PP;
  PrintingPolicy printPol;

  uint64_t NumInstances;
  uint64_t HashNum;
  bool CreatesTasks = false;
  bool UsesOmpif = false;
  bool MemcpyWideport = false;
  bool UsesLock = false;
  WrapperPortMap WrapperPortMap;

  llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo> Localmems;

  Optional<uint64_t> getNumInstances() {
    uint64_t value = 1; // Default is 1 instance
    if (auto *numInstances =
            FD->getAttr<OSSTaskDeclAttr>()->getNumInstances()) {
      if (auto number = numInstances->getIntegerConstantExpr(SourceContext);
          number && *number > 0) {
        value = number->getZExtValue();
      } else {
        Diag(numInstances->getExprLoc(),
             diag::err_expected_constant_unsigned_integer);
        return llvm::NoneType{};
      }
    }

    return value;
  }

  Optional<llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo>>
  ComputeLocalmems() {
    auto *taskAttr = FD->getAttr<OSSTaskDeclAttr>();
    bool foundError = false;
    // First, compute the direction tags of the parameters. Do note that not
    // all parameters are guaranteed to be present
    llvm::SmallDenseMap<
        const ParmVarDecl *,
        std::pair<const OSSArrayShapingExpr *, LocalmemInfo::Dir>>
        currentAssignationsOfArrays;
    auto EmitDepListIterDecls = [&](auto &&DepExprsIter,
                                    LocalmemInfo::Dir dir) {
      for (const Expr *DepExpr : DepExprsIter) {
        auto *arrShapingExpr = dyn_cast<OSSArrayShapingExpr>(DepExpr);
        if (!arrShapingExpr)
          return;
        auto *arrExprBase = dyn_cast<DeclRefExpr>(
            arrShapingExpr->getBase()->IgnoreParenImpCasts());
        assert(arrExprBase);
        auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
        assert(decl);
        auto res = currentAssignationsOfArrays.find(decl);
        if (res != currentAssignationsOfArrays.end()) {
          res->second.second = LocalmemInfo::Dir(res->second.second | dir);
        } else {
          currentAssignationsOfArrays.insert({decl, {arrShapingExpr, dir}});
        };
      }
    };
    EmitDepListIterDecls(taskAttr->ins(), LocalmemInfo::IN);
    EmitDepListIterDecls(taskAttr->outs(), LocalmemInfo::OUT);
    EmitDepListIterDecls(taskAttr->inouts(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->concurrents(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->weakIns(), LocalmemInfo::IN);
    EmitDepListIterDecls(taskAttr->weakOuts(), LocalmemInfo::OUT);
    EmitDepListIterDecls(taskAttr->weakInouts(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->weakConcurrents(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->depIns(), LocalmemInfo::IN);
    EmitDepListIterDecls(taskAttr->depOuts(), LocalmemInfo::OUT);
    EmitDepListIterDecls(taskAttr->depInouts(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->depConcurrents(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->depWeakIns(), LocalmemInfo::IN);
    EmitDepListIterDecls(taskAttr->depWeakOuts(), LocalmemInfo::OUT);
    EmitDepListIterDecls(taskAttr->depWeakInouts(), LocalmemInfo::INOUT);
    EmitDepListIterDecls(taskAttr->depWeakConcurrents(), LocalmemInfo::INOUT);

    // Then compute the list of localmem parameters
    llvm::SmallDenseSet<const ParmVarDecl *> parametersToLocalmem;

    // If copy_deps
    if (taskAttr->getCopyDeps()) {
      for (auto *param : FD->parameters()) {
        if (currentAssignationsOfArrays.find(param) !=
            currentAssignationsOfArrays.end()) {
          parametersToLocalmem.insert(param);
        }
      }
    }
    // If we have an explicit list of localmem (copy_in, copy_out, copy_inout),
    // use that
    auto explicitCopy = [&](auto &&list, LocalmemInfo::Dir dir) {
      for (auto *localmem : list) {
        auto *arrShapingExpr = dyn_cast<OSSArrayShapingExpr>(localmem);
        if (!arrShapingExpr) {
          Diag(localmem->getExprLoc(),
               diag::err_oss_fpga_expected_array_to_place_in_localmem);
          foundError = true;
          continue;
        }

        auto *arrExprBase = dyn_cast<DeclRefExpr>(
            arrShapingExpr->getBase()->IgnoreParenImpCasts());
        assert(arrExprBase);
        auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
        assert(decl);
        parametersToLocalmem.insert(decl);
        auto &def = currentAssignationsOfArrays[decl];
        def.first = arrShapingExpr;
        def.second = LocalmemInfo::Dir(def.second | dir);
      }
    };
    explicitCopy(taskAttr->copyIn(), LocalmemInfo::IN);
    explicitCopy(taskAttr->copyOut(), LocalmemInfo::OUT);
    explicitCopy(taskAttr->copyInOut(), LocalmemInfo::INOUT);

    // Now check that none of the decl are const qualified while out, and that
    // we know the sizes
    for (auto *param : parametersToLocalmem) {
      if (currentAssignationsOfArrays.find(param)->second.second &
              LocalmemInfo::OUT &&
          param->getType()->getPointeeType().isConstQualified()) {
        Diag(
            param->getLocation(),
            diag::
                err_oss_fpga_param_used_in_localmem_marked_as_out_const_qualified);
        foundError = true;
      }
      for (auto *shape :
           currentAssignationsOfArrays.find(param)->second.first->getShapes()) {
        if (auto valInteger = shape->getIntegerConstantExpr(SourceContext);
            !valInteger || *valInteger < 0) {
          Diag(shape->getExprLoc(),
               diag::err_expected_constant_unsigned_integer);
          foundError = true;
        }
      }
      auto paramType = param->getType();
      auto numShapes = currentAssignationsOfArrays.find(param)
                           ->second.first->getShapes()
                           .size();
      for (size_t i = 0; i < numShapes; ++i) {
        paramType = DerefOnceTypePointerTo(paramType);
      }
      while (paramType->isArrayType()) {
        if (!paramType->isConstantArrayType()) {
          Diag(param->getLocation(),
               diag::err_expected_constant_unsigned_integer);
        }
        paramType = DerefOnceTypePointerTo(paramType);
      }
      if (paramType->isPointerType()) {
        Diag(param->getLocation(),
             diag::err_expected_constant_unsigned_integer);
      }
    }
    if (foundError) {
      return llvm::NoneType{};
    }
    // Compute the localmem list
    llvm::SmallDenseMap<const ParmVarDecl *, LocalmemInfo> localmemList;
    for (auto *param : parametersToLocalmem) {
      auto data = currentAssignationsOfArrays.find(param);
      localmemList.insert(
          {param, LocalmemInfo{-1, data->second.first, data->second.second}});
    }
    return localmemList;
  }

  Optional<uint64_t> GenOnto() {
    auto *taskAttr = FD->getAttr<OSSTaskDeclAttr>();
    auto *ontoExpr = taskAttr->getOnto();
    // Check onto information
    if (ontoExpr) {
      auto ontoRes = ontoExpr->getIntegerConstantExpr(SourceContext);
      if (!ontoRes || *ontoRes < 0) {
        Diag(ontoExpr->getExprLoc(),
             diag::err_expected_constant_unsigned_integer);
        return llvm::NoneType{};
      }
      uint64_t onto = ontoRes->getZExtValue();
      // Check that arch bits are set
      if ((onto & 0x300000000) == 0) {
        Diag(ontoExpr->getExprLoc(),
             diag::err_oss_fpga_onto_clause_missing_bits);
      } else if (onto > 0x3FFFFFFFF) {
        Diag(ontoExpr->getExprLoc(),
             diag::err_oss_fpga_onto_clause_task_type_too_wide);
      }
      return onto;
    }
    auto simpleHashStr = [](const char *str) {
      const int MULTIPLIER = 33;
      unsigned int h;
      unsigned const char *p;

      h = 0;
      for (p = (unsigned const char *)str; *p != '\0'; p++)
        h = MULTIPLIER * h + *p;

      h += (h >> 5);

      return h; // or, h % ARRAY_SIZE;
    };

    // Not using the line number to allow future modifications of source code
    // without afecting the accelerator hash
    std::string typeStr;
    llvm::raw_string_ostream typeStream(typeStr);

    typeStream << SourceMgr.getFilename(FD->getSourceRange().getBegin()) << " "
               << OrigFuncName;
    unsigned long long int type = simpleHashStr(typeStr.c_str()) &
                                  0xFFFFFFFF; //< Ensure that it its upto 32b
    // FPGA flag
    type |= 0x100000000;
    return type;
  }

  void generateMemcpyWideportFunction(bool in) {

    const std::string memPtrType =
        "ap_uint<" +
        std::to_string(SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth) + ">";
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
    if (SemaRef.getLangOpts().OmpSsFpgaCheckLimitsMemoryPort) {
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
      if (SemaRef.getLangOpts().OmpSsFpgaCheckLimitsMemoryPort) {
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
    OutputHeaders << R"(
///////////////////
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
    Output << R"(
struct mcxx_outaxis {
  ap_uint<64> data;
  ap_uint<2> dest;
  ap_uint<1> last;
};
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

    if (WrapperPortMap[FD][size_t(WrapperPort::MEMORY_PORT)]) {
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
  }

  bool GenOriginalFunctionMoved() {
    // Dependency resolution. We use the ASTImporter utility, which is able to
    // manage any sort of C++ construct during resolution.
    DiagnosticsEngine toDiagnostics(new DiagnosticIDs(),
                                    new DiagnosticOptions());
    FileManager fileManager(SourceMgr.getFileManager().getFileSystemOpts());
    SourceManager toMgr(toDiagnostics, fileManager);
    LangOptions toLangOpts(SourceContext.getLangOpts());
    IdentifierTable toIdentifierTable;
    SelectorTable toSelectorTable;
    ASTContext toContext(toLangOpts, toMgr, toIdentifierTable, toSelectorTable,
                         PP.getBuiltinInfo(), TU_Incremental);
    toContext.InitBuiltinTypes(SourceContext.getTargetInfo());

    ASTImporter importer(toContext, toMgr.getFileManager(), SourceContext,
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
    (*importedOrErr)->dropAttr<OSSTaskDeclAttr>();

    // Headers
    // We are only going to preserve the headers ending with .fpga.h or .fpga,
    // this was a restriction in the original ompss@fpga and it simplifies some
    // tasks in a major way.
    for (auto &&file : PP.getIncludedFiles()) {
      if (!file->getName().endswith(".fpga.h") &&
          !file->getName().endswith(".fpga")) {
        continue;
      }
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

      llvm::StringRef headerInclude(data, dataEndInclude - data);
      OutputHeaders << "#include " << headerInclude << '\n';
    }

    // Body functions
    for (Decl *otherDecl : toContext.getTranslationUnitDecl()->decls()) {
      if (SourceMgr.getFileID(otherDecl->getSourceRange().getBegin()) !=
          SourceMgr.getFileID(FD->getSourceRange().getBegin())) {
        // Skip dependency not originating in the file.
        continue;
      }

      if (auto *funcDecl = dyn_cast<FunctionDecl>(otherDecl);
          funcDecl && !funcDecl->hasBody()) {
        Diag(otherDecl->getLocation(),
             diag::err_oss_fpga_missing_body_for_function_depended_by_kernel);
        Diag(FD->getLocation(), diag::note_oss_fpga_kernel);
        return false;
      } else if (funcDecl) {
        auto origName = funcDecl->getDeclName();
        auto &id =
            PP.getIdentifierTable().get(origName.getAsString() + "_moved");

        DeclarationName name(&id);
        funcDecl->setDeclName(name);
      }
    }

    for (Decl *otherDecl : toContext.getTranslationUnitDecl()->decls()) {
      if (SourceMgr.getFileID(otherDecl->getSourceRange().getBegin()) !=
          SourceMgr.getFileID(FD->getSourceRange().getBegin())) {
        // Skip dependency not originating in the file.
        continue;
      }

      otherDecl->print(Output, 0, true);
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
    auto *varDecl = VarDecl::Create(
        SourceContext, SemaRef.getCurLexicalContext(), SourceLocation(),
        SourceLocation(), &id, type, nullptr, storageClass);

    std::string res;
    llvm::raw_string_ostream stream(res);
    varDecl->print(stream);
    return res;
  }

  QualType DerefOnceTypePointerTo(QualType type) {

    if (type->isPointerType()) {
      return type->getPointeeType().IgnoreParens();
    }
    if (type->isArrayType()) {
      return type->getAsArrayTypeUnsafe()->getElementType().IgnoreParens();
    }
    return type;
  }

  QualType GetElementTypePointerTo(QualType type) {
    QualType pointsTo = type;

    for (auto isPointer = pointsTo->isPointerType(),
              isArray = pointsTo->isArrayType();
         isPointer || isArray; isPointer = pointsTo->isPointerType(),
              isArray = pointsTo->isArrayType()) {
      if (isPointer) {
        pointsTo = pointsTo->getPointeeType().IgnoreParens();
      } else if (isArray) {
        pointsTo =
            pointsTo->getAsArrayTypeUnsafe()->getElementType().IgnoreParens();
      }
    }

    return pointsTo;
  }

  QualType LocalmemArrayType(const OSSArrayShapingExpr *arrayType) {
    auto paramType = arrayType->getBase()->getType();
    for (size_t i = 0; i < arrayType->getShapes().size(); ++i)
      paramType = DerefOnceTypePointerTo(paramType);
    for (auto *shape : arrayType->getShapes()) {
      auto computedSize = shape->getIntegerConstantExpr(SourceContext);
      if (!computedSize) {
        llvm_unreachable(
            "We have already checked that the shape expressions evaluate to "
            "positive integers, we should be able to use them here safely");
      }
      paramType = SourceContext.getConstantArrayType(
          paramType, *computedSize, shape, ArrayType::ArraySizeModifier{}, 0);
    }
    return paramType;
  }

  uint64_t ComputeArrayRefSize(const OSSArrayShapingExpr *arrayType,
                               uint64_t baseType = 1) {
    uint64_t totalSize = baseType;
    auto paramType = LocalmemArrayType(arrayType);
    while (paramType->isArrayType()) {
      if (!paramType->isConstantArrayType()) {
        llvm_unreachable("We have already checked that the parameter type is a "
                         "constant array");
      }
      auto *arrType = dyn_cast<ConstantArrayType>(paramType);
      paramType = DerefOnceTypePointerTo(paramType);
      totalSize *= arrType->getSize().getZExtValue();
    }

    return totalSize;
  }

  void GenerateWrapperFunctionLocalmems() {
    for (auto &&p : Localmems) {
      Output << "  "
             << GetDeclVariableString(StorageClass::SC_Static,
                                      LocalmemArrayType(p.second.FixedArrayRef),
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
      auto *it = WrapperPortMap.find(FD);
      return it != WrapperPortMap.end() &&
             it->second[(int)WrapperPort::MEMORY_PORT];
    }();

    if (!CreatesTasks) {
      if ((SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth > 0 &&
           Localmems.size() > 0) ||
          forceMemport) {
        Output << ", ap_uint<" << SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth
               << ">* mcxx_memport";
      }

      for (auto *param : FD->parameters()) {
        auto it = Localmems.find(param);
        if (SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth > 0 &&
            it != Localmems.end())
          continue;

        QualType paramType = param->getType();
        if (paramType->isPointerType() || paramType->isArrayType()) {
          Output << ", "
                 << GetDeclVariableString(
                        StorageClass::SC_None,
                        SourceContext.getPointerType(
                            GetElementTypePointerTo(paramType)),
                        " mcxx_" + param->getNameAsString());
        }
      }
    }
    if (UsesOmpif) {
      Output << ", unsigned char ompif_rank, unsigned char ompif_size";
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
      if ((SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth > 0 &&
           Localmems.size() > 0) ||
          forceMemport) {
        Output << "#pragma HLS interface m_axi port=mcxx_memport\n";
      }
      for (auto *param : FD->parameters()) {
        auto it = Localmems.find(param);
        if (SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth > 0 &&
            it != Localmems.end())
          continue;

        QualType paramType = param->getType();
        if (paramType->isPointerType() || paramType->isArrayType()) {
          Output << "#pragma HLS interface m_axi port=mcxx_" << param->getName()
                 << "\n";
        }
      }
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
            SourceContext.getPointerType(GetElementTypePointerTo(paramType));
      }
      if (!usesMemoryPort && paramType.isConstQualified()) {
        paramType.removeLocalConst();
      }
      return std::pair{paramType, usesMemoryPort};
    };
    auto paramId = 0;
    for (auto *param : FD->parameters()) {
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
            paramType = SourceContext.getPointerType(
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
    for (auto *param : FD->parameters()) {
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
    for (auto &&[param, localmemInfo] : Localmems) {
      if ((localmemInfo.dir & dir) == 0) {
        continue;
      }

      const auto &fixedArrayRef = localmemInfo.FixedArrayRef;
      int paramId = localmemInfo.ParamIdx;
      QualType baseType = GetElementTypePointerTo(param->getType());
      uint64_t baseTypeSize =
          SourceContext.getTypeSize(baseType) / SourceContext.getCharWidth();

      const auto paramName = param->getName();
      const auto baseTypeSizeStr = std::to_string(baseTypeSize);
      const std::string memPtrType =
          "ap_uint<" +
          (SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth > 0
               ? std::to_string(SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth)
               : std::to_string(baseTypeSize * 8)) +
          ">";
      const std::string sizeofMemPtrType = "sizeof(" + memPtrType + ")";

      const std::string dataReferenceSize =
          "(" +
          std::to_string(ComputeArrayRefSize(fixedArrayRef, baseTypeSize)) +
          ")";
      const std::string nElementsSrc =
          "(" + dataReferenceSize + "/" + baseTypeSizeStr + ")";
      const std::string nElemsRead =
          "(sizeof(" + memPtrType + ")/" + baseTypeSizeStr + ")";

      Output << "  if (mcxx_flags_" << paramId << "["
             << (dir == LocalmemInfo::IN ? STR_ARG_FLAG_IN_BIT
                                         : STR_ARG_FLAG_OUT_BIT)
             << "]) {\n";
      if (SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth == 0) {
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
        if (SemaRef.getLangOpts().OmpSsFpgaCheckLimitsMemoryPort) {
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
          if (SemaRef.getLangOpts().OmpSsFpgaCheckLimitsMemoryPort) {
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
  }

  void GenerateWrapperFunctionUserTaskCall() {
    auto &outs = Output << "  " << TaskFuncName << "(";
    bool first = true;
    auto printSeparator = [&]() {
      if (first) {
        first = false;
      } else {
        outs << ", ";
      }
    };

    for (auto *param : FD->parameters()) {
      printSeparator();
      outs << param->getName();
    }
    auto *it = WrapperPortMap.find(FD);
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
    outs << ");\n";
  }

  void GenerateWrapperFunction() {
    if (SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth == 0 &&
        !Localmems.empty()) {
      OutputHeaders << "#include <string.h> //needed for memcpy\n";
    }
    Output << "void mcxx_write_out_port(const ap_uint<64> data, const "
              "ap_uint<3> dest, const ap_uint<1> last, " STR_OUTPORT_DECL
              ") {\n";
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

    Output << "  " STR_INPORT_READ_NODATA "; //command word\n";
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
  }

public:
  WrapperGenerator(Sema &SemaRef, Callable Diag, llvm::raw_ostream &OutputFile,
                   FunctionDecl *FD, llvm::StringRef FuncName,
                   ASTContext &SourceContext, SourceManager &SourceMgr,
                   Preprocessor &PP)
      : SemaRef(SemaRef), Diag(std::forward<Callable>(Diag)),
        OutputHeaders(OutputStrHeaders), Output(OutputStr),
        OutputFinalFile(OutputFile), FD(FD), OrigFuncName(FuncName),
        TaskFuncName(std::string(FuncName) + "_moved"),
        SourceContext(SourceContext), SourceMgr(SourceMgr), PP(PP),
        printPol(SemaRef.getLangOpts()) {}

  bool GenerateWrapperFile() {
    auto numInstances = getNumInstances();
    if (!numInstances) {
      return false;
    }
    NumInstances = std::move(*numInstances);

    auto localmems = ComputeLocalmems();
    if (!localmems) {
      return false;
    }
    Localmems = std::move(*localmems);

    auto hashNum = GenOnto();
    if (!hashNum) {
      return false;
    }
    HashNum = std::move(*hashNum);

    FPGAFunctionTreeVisitor visitor(FD, WrapperPortMap);
    visitor.TraverseStmt(FD->getBody());
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
};
} // namespace

bool Sema::ActOnOmpSsFpgaExtractFiles() {
  if (ompssFpgaDecls.empty())
    return true;

  auto diag = [&](auto... arg) { return Diag(arg...); };
  auto realPathOrNone =
      getAbsoluteDirExport(SourceMgr, getLangOpts().OmpSsFpgaHlsTasksDir,
                           ompssFpgaDecls[0]->getLocation(), diag);
  if (!realPathOrNone) {
    return false;
  }

  for (auto *decl : ompssFpgaDecls) {
    auto *FD = dyn_cast<FunctionDecl>(decl);

    auto funcName = FD->getName();
    std::ofstream stream{*realPathOrNone + "/" +
                         (funcName.str() + "_hls_automatic_clang.cpp")};
    llvm::raw_os_ostream outputFile(stream);

    if (!GenerateExtractedOriginalFunction(diag, outputFile, FD, funcName,
                                           Context, SourceMgr, PP)) {
      return false;
    }
  }
  return true;
}

bool Sema::ActOnOmpSsFpgaGenerateWrapperCodeFiles() {
  if (ompssFpgaDecls.empty())
    return true;

  auto diag = [&](auto... arg) { return Diag(arg...); };
  auto realPathOrNone =
      getAbsoluteDirExport(SourceMgr, getLangOpts().OmpSsFpgaHlsTasksDir,
                           ompssFpgaDecls[0]->getLocation(), diag);
  if (!realPathOrNone) {
    return false;
  }

  for (auto *decl : ompssFpgaDecls) {
    auto *FD = dyn_cast<FunctionDecl>(decl);

    auto funcName = FD->getName();
    std::ofstream stream{*realPathOrNone + "/" +
                         (funcName.str() + "_hls_automatic_clang.cpp")};
    llvm::raw_os_ostream outputFile(stream);

    if (!WrapperGenerator<decltype(diag)>(*this, diag, outputFile, FD, funcName,
                                          Context, SourceMgr, PP)
             .GenerateWrapperFile()) {
      return false;
    }
  }
  return true;
}

bool Sema::ActOnOmpSsDeclareTaskDirectiveWithFpga(Decl *ADecl) {
  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_oss_function_expected);
    return false;
  }

  if (!FD->doesThisDeclarationHaveABody()) {
    Diag(ADecl->getLocation(), diag::err_oss_function_with_body_expected);
    return false;
  }

  ompssFpgaDecls.push_back(ADecl);
  return true;
}