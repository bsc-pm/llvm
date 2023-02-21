//===--- OmpssFpgaWrapperGen.h - OmpSs FPGA wrapper generator ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the Frontend pass to generate the wrapper file to be sent to the
/// fpga
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_OMPSSFPGAWRAPPERGEN_H
#define LLVM_CLANG_FRONTEND_OMPSSFPGAWRAPPERGEN_H

#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

class FPGAWrapperGen : public clang::ASTConsumer {
private:
  clang::Preprocessor &PP;
  void ActOnOmpSsFpgaExtractFiles(clang::ASTContext &Ctx);
  void ActOnOmpSsFpgaGenerateWrapperCodeFiles(clang::ASTContext &Ctx);

public:
  FPGAWrapperGen(clang::Preprocessor &PP);
  ~FPGAWrapperGen() override;
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};
#endif