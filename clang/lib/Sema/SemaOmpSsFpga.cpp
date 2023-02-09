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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
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
  int param_idx = -1;
  const OSSArrayShapingExpr *fixed_array_ref;
  enum Dir { IN = 0b01, OUT = 0b10, INOUT = 0b11 };
  Dir dir;
};

template <typename Callable>
Optional<std::filesystem::path>
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
  return std::filesystem::u8path(realPathStr.begin(), realPathStr.end());
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
  std::string OutputStr;
  llvm::raw_string_ostream Output;
  llvm::raw_ostream &OutputFinalFile;
  FunctionDecl *FD;
  llvm::StringRef OrigFuncName;
  std::string TaskFuncName;
  ASTContext &SourceContext;
  SourceManager &SourceMgr;
  Preprocessor &PP;

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

    // If explicit localmem_copies or no localmem() attr and no
    // no_localmem_copies, We copy all the arrays to localmem
    if (taskAttr->getLocalmemCopies() ||
        (taskAttr->localmem_size() == 0 && !taskAttr->getNoLocalmemCopies())) {
      for (auto *param : parametersToLocalmem) {
        if (currentAssignationsOfArrays.find(param) !=
            currentAssignationsOfArrays.end()) {
          parametersToLocalmem.insert(param);
        }
      }
    }
    // If we have an explicit list of localmem, use that
    else if (taskAttr->localmem_size() > 0) {
      for (auto *localmem : taskAttr->localmem()) {
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
        // Could not find the direction of the array? Assume In/Out if not
        // const, otherwise, In
        if (currentAssignationsOfArrays.find(decl) ==
            currentAssignationsOfArrays.end()) {
          if (decl->getType()->getPointeeType().isConstQualified()) {
            currentAssignationsOfArrays.insert(
                {decl, {arrShapingExpr, LocalmemInfo::IN}});
          } else {
            currentAssignationsOfArrays.insert(
                {decl, {arrShapingExpr, LocalmemInfo::INOUT}});
          }
        }
      }
    }

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
    /*
    const std::string mem_ptr_type =
        "ap_uint<" + fpga_options.memory_port_width + ">";
    const std::string sizeof_mem_ptr_type = "sizeof(" + mem_ptr_type + ")";
    const std::string n_elems_read = "(sizeof(" + mem_ptr_type + ")/sizeof(T))";

    os() << "template<class T>";
    os()
        << "void nanos6_fpga_memcpy_wideport_" << (in ? "in" : "out") << "(T * "
        << (in ? "dst" : "src")
        << ", const unsigned long long int addr, const unsigned int num_elems, "
        << mem_ptr_type << "* mcxx_memport) {";
    os() << "#pragma HLS inline";
    push_indent();
    os() << "for (int i = 0; i < (num_elems-1)/" << n_elems_read
         << "+1; ++i) {";
    os() << "#pragma HLS pipeline II=1";
    push_indent();
    os() << mem_ptr_type << " tmpBuffer;";
    if (in)
      os() << "tmpBuffer = *(mcxx_memport + addr/" << sizeof_mem_ptr_type
           << " + i);";
    os() << "for (int j = 0; j < " << n_elems_read << "; ++j) {";
    push_indent();
    if (fpga_options.check_limits_memory_port) {
      os() << "if (i*" << n_elems_read << "+j >= num_elems)";
      push_indent();
      os() << "break;";
      pop_indent();
    }
    os() << "__mcxx_cast<T> cast_tmp;";
    if (in) {
      os() << "cast_tmp.raw = tmpBuffer((j+1)*sizeof(T)*8-1, j*sizeof(T)*8);";
      os() << "dst[i*" << n_elems_read << "+j] = cast_tmp.typed;";
    } else {
      os() << "cast_tmp.typed = src[i*" << n_elems_read << "+j];";
      os() << "tmpBuffer((j+1)*sizeof(T)*8-1, j*sizeof(T)*8) = cast_tmp.raw;";
    }
    pop_indent();
    os() << "}";
    if (!in) {
      if (fpga_options.check_limits_memory_port) {
        os() << "const int rem = num_elems-(i*" << n_elems_read << ");";
        os() << "const unsigned int bit_l = 0;";
        os() << "const unsigned int bit_h = rem >= " << n_elems_read << " ? ("
             << sizeof_mem_ptr_type << "*8-1) : ((rem*sizeof(T))*8-1);";
        os() << "mcxx_memport[addr/sizeof(" << mem_ptr_type << ") + i]"
             << "(bit_h, bit_l) = tmpBuffer(bit_h, bit_l);";
      } else {
        os() << "*(mcxx_memport + addr/" << sizeof_mem_ptr_type
             << " + i) = tmpBuffer;";
      }
    }
    pop_indent();
    os() << "}";
    pop_indent();
    os() << "}";
    */
  }

  void GenerateWrapperHeader() {
    Output << R"(
///////////////////
// Automatic IP Generated by OmpSs@FPGA compiler
///////////////////
// The below code is composed by:
//  1) User source code, which may be under any license (see in original source code)
//  2) OmpSs@FPGA toolchain code which is licensed under LGPLv3 terms and conditions
///////////////////
)";
    Output << "// Top IP Function: " << TaskFuncName << '\n';
    Output << "// Accel. type hash: " << HashNum << '\n';
    Output << "// Num. instances: " << NumInstances << '\n';
    Output << "// Wrapper version: " << WrapperVersion << '\n';
    Output << "///////////////////" << '\n';
  }

  void GenerateWrapperTop() {
    Output << R"#(
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
             "copies[], " STR_OUTPORT_DECL ");";

      Output << "void mcxx_taskwait(" STR_SPWNINPORT_DECL ", " STR_OUTPORT_DECL
                ");";
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

    std::vector<std::string> renamedNames;
    // Body functions
    for (Decl *otherDecl : toContext.getTranslationUnitDecl()->decls()) {
      if (auto *funcDecl = dyn_cast<FunctionDecl>(otherDecl);
          funcDecl && !funcDecl->hasBody()) {
        Diag(otherDecl->getLocation(),
             diag::err_oss_fpga_missing_body_for_function_depended_by_kernel);
        Diag(FD->getLocation(), diag::note_oss_fpga_kernel);
        return false;
      } else if (funcDecl) {
        auto origName = funcDecl->getDeclName();
        renamedNames.push_back(origName.getAsString() + "_moved");
        auto &id = PP.getIdentifierTable().get(renamedNames.back());

        DeclarationName name(&id);
        funcDecl->setDeclName(name);
      }
    }
    toContext.getTranslationUnitDecl()->print(Output, 0, true);
    return true;
  }

  void GenerateWrapperFunction() {
    if (SemaRef.getLangOpts().OmpSsFpgaMemoryPortWidth == 0 &&
        !Localmems.empty()) {
      Output << "#include <string.h> //needed for memcpy\n";
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
    /*
    generate_wrapper_function_params();
    push_indent();
    generate_wrapper_function_localmems();
    os() << STR_INPORT_READ_NODATA "; //command word";
    os() << STR_TASKID " = " STR_INPORT_READ ";";
    os() << "ap_uint<64> " << STR_PARENT_TASKID " = " STR_INPORT_READ ";";
    generate_wrapper_function_param_reads();
    generate_wrapper_function_localmem_copies(LocalmemInfo::IN);
    generate_wrapper_function_user_task_call();
    generate_wrapper_function_localmem_copies(LocalmemInfo::OUT);
    os() << "{"; // send finish task
    push_indent();
    os() << "#pragma HLS protocol fixed";
    os() << "ap_uint<64> header = 0x03;";
    os() << "ap_wait();";
    os() << STR_OUTPORT_WRITE_FUN("header, " STR_FINISH_TASK_CODE ", 0");
    os() << "ap_wait();";
    os() << STR_OUTPORT_WRITE_FUN(STR_TASKID ", " STR_FINISH_TASK_CODE ", 0");
    os() << "ap_wait();";
    os() << STR_OUTPORT_WRITE_FUN(STR_PARENT_TASKID ", " STR_FINISH_TASK_CODE
                                                    ", 1");
    os() << "ap_wait();";
    pop_indent();
    os() << "}";
    pop_indent();
    os() << "}";
    */
  }

public:
  WrapperGenerator(Sema &SemaRef, Callable Diag, llvm::raw_ostream &OutputFile,
                   FunctionDecl *FD, llvm::StringRef FuncName,
                   ASTContext &SourceContext, SourceManager &SourceMgr,
                   Preprocessor &PP)
      : SemaRef(SemaRef), Diag(std::forward<Callable>(Diag)), Output(OutputStr),
        OutputFinalFile(OutputFile), FD(FD), OrigFuncName(FuncName),
        TaskFuncName(std::string(FuncName) + "_moved"),
        SourceContext(SourceContext), SourceMgr(SourceMgr), PP(PP) {}

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
    std::ofstream stream{*realPathOrNone /
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
    std::ofstream stream{*realPathOrNone /
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