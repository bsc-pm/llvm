//===--- OmpssFpgaWrapperGen.h - OmpSs FPGA wrapper generator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

class FPGAWrapperGen : public clang::ASTConsumer {
private:
  clang::Preprocessor &PP;
  clang::CompilerInstance &CI;
  void ActOnOmpSsFpgaExtractFiles(clang::ASTContext &Ctx);
  void ActOnOmpSsFpgaGenerateWrapperCodeFiles(clang::ASTContext &Ctx);

public:
  FPGAWrapperGen(clang::Preprocessor &PP, clang::CompilerInstance &CI);
  ~FPGAWrapperGen() override;
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};
#endif