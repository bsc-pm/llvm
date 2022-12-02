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
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOmpSs.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <fstream>
#include <utility>

using namespace clang;

namespace {
template <class Callable>
bool GenerateHeaderIncludeBlock(Callable &&Diag, llvm::raw_ostream &outputFile,
                                Preprocessor &PP, SourceManager &SourceMgr,
                                SourceLocation const &functionStart) {
  // We are only interested in the included files in the file where the function
  // is defined. I don't want to include more than what's necessary, as I don't
  // know what macro shenanigans might be going on.
  //
  // This approach isn't optimal, as superfluous includes are not preserved, and
  // this could introduce differences when we recompile. I Don't have a perfect
  // solution for this yet. TODO: Find a better way?
  //
  // This approach can also be wrong, in the case the function is in a header
  // that depends that a previous header is included.
  for (auto &&file : PP.getIncludedFiles()) {
    auto loc = SourceMgr.getIncludeLoc(SourceMgr.translateFile(file));
    if (SourceMgr.getFileID(loc) != SourceMgr.getFileID(functionStart)) {
      // Skip include not included in the file of the function.
      continue;
    }
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
  return true;
}

template <class Callable>
bool GenerateOriginalFuncionBody(Callable &&Diag, llvm::raw_ostream &outputFile,
                                 FunctionDecl *FD, llvm::StringRef funcName,
                                 SourceLocation start, SourceLocation end,
                                 SourceManager &SourceMgr) {

  // The current approach is very much ad-hoc. Once we add support for
  // reprinting the #pragma HLS directives, we should switch to using the AST
  // itself

  auto [startOffset, endOffset] =
      std::pair{SourceMgr.getFileOffset(start), SourceMgr.getFileOffset(end)};
  auto nameFunctionStart =
      SourceMgr.getFileOffset(FD->getNameInfo().getSourceRange().getBegin());
  auto *funcData = SourceMgr.getCharacterData(start);
  llvm::StringRef stringFunction(funcData, endOffset - startOffset + 1);

  auto locationName = stringFunction.find(
      funcName,
      std::max(int64_t(nameFunctionStart) - int64_t(startOffset), 0l));
  if (locationName != std::string::npos) {
    outputFile << stringFunction.substr(0, locationName + funcName.size())
               << "_moved"
               << stringFunction.substr(locationName + funcName.size());
  } else {
    Diag(FD->getLocation(), diag::err_oss_fpga_transform_broken);
    return false;
  }

  return true;
}
} // namespace

bool Sema::ActOnOmpSsDeclareTaskDirectiveWithFpga(Decl *ADecl) {
  auto diag = [&](auto... arg) { Diag(arg...); };
  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_oss_function_expected);
    return false;
  }

  if (!FD->doesThisDeclarationHaveABody()) {
    Diag(ADecl->getLocation(), diag::err_oss_function_with_body_expected);
    return false;
  }

  auto funcName = FD->getName();
  std::ofstream stream{funcName.str() + "_hls_automatic_clang.cpp"};
  llvm::raw_os_ostream outputFile(stream);

  auto range = ADecl->getSourceRange();
  auto [start, end] = std::pair{range.getBegin(), range.getEnd()};

  return GenerateHeaderIncludeBlock(diag, outputFile, PP, SourceMgr, start) &&
         GenerateOriginalFuncionBody(diag, outputFile, FD, funcName, start, end,
                                     SourceMgr);
}