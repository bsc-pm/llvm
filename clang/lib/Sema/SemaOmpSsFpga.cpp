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
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
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
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>

using namespace clang;

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

namespace {
template <class Callable>
bool GenerateFunctionBody(Callable &&Diag, llvm::raw_ostream &outputFile,
                          FunctionDecl *FD, llvm::StringRef funcName,
                          SourceLocation start, SourceLocation end,
                          ASTContext &sourceContext, SourceManager &SourceMgr,
                          Preprocessor &PP) {
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

  // Extra Headers needed. No need to bother checking for repeated includes.
  outputFile << R"(#include <systemc.h>
#include <ap_int.h>
#include <cstring>
#include <hls_stream.h>
)";

  // Body functions
  std::string out;
  llvm::raw_string_ostream stream(out);
  for (Decl *decl : toContext.getTranslationUnitDecl()->decls()) {
    if (SourceMgr.getFileID(decl->getSourceRange().getBegin()) !=
        SourceMgr.getFileID(start)) {
      // Skip dependency not originating in the file.
      continue;
    }

    if (dyn_cast<FunctionDecl>(decl) && !decl->hasBody()) {
      Diag(decl->getLocation(),
           diag::err_oss_fpga_missing_body_for_function_depended_by_kernel);
      Diag(FD->getLocation(), diag::note_oss_fpga_kernel);
      return false;
    }

    decl->print(stream, 0, true);
    if (!out.empty() && out[out.size() - 1] != '\n') {
      stream << ";\n"; // This may lead to superfluous ';', but I'm not
                       // quite sure how to prevent them easily.
                       // Trailing ; are legal fortunately!
    }
  }
  outputFile << out;

  return true;
}
} // namespace

bool Sema::ActOnOmpSsFpgaGenerateAitFiles() {
  if (ompssFpgaDecls.empty())
    return true;

  auto &vfs = SourceMgr.getFileManager().getVirtualFileSystem();
  std::string path = getLangOpts().OmpSsFpgaExtractHlsTasksDir;
  if (path.empty()) {
    Diag(ompssFpgaDecls[0]->getLocation(),
         diag::err_oss_ompss_fpga_extract_hls_tasks_dir_missing_dir);
    return false;
  }
  llvm::SmallVector<char, 128> realPathStr;
  if (vfs.getRealPath(path, realPathStr)) {
    Diag(ompssFpgaDecls[0]->getLocation(),
         diag::err_oss_ompss_fpga_extract_hls_tasks_dir_missing_dir);
    return false;
  }
  std::filesystem::path realPath =
      std::filesystem::u8path(realPathStr.begin(), realPathStr.end());

  auto diag = [&](auto... arg) { return Diag(arg...); };
  for (auto *decl : ompssFpgaDecls) {
    auto *FD = dyn_cast<FunctionDecl>(decl);

    auto funcName = FD->getName();
    std::ofstream stream{realPath /
                         (funcName.str() + "_hls_automatic_clang.cpp")};
    llvm::raw_os_ostream outputFile(stream);

    auto range = decl->getSourceRange();
    auto [start, end] = std::pair{range.getBegin(), range.getEnd()};
    if (!GenerateFunctionBody(diag, outputFile, FD, funcName, start, end,
                              Context, SourceMgr, PP)) {
      return false;
    }
  }
  return true;
}