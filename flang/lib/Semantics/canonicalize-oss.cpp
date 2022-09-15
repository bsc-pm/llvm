//===-- lib/Semantics/canonicalize-oss.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonicalize-oss.h"
#include "flang/Parser/parse-tree-visitor.h"

// After Loop Canonicalization, rewrite OmpSs-2 parse tree to make OmpSs-2
// Constructs more structured which provide explicit scopes for later
// structural checks and semantic analysis.
//   1. move structured DoConstruct and OSSEndLoopDirective into
//      OmpSsLoopConstruct. Cossilation will not proceed in case of errors
//      after this pass.
//   2. TBD
namespace Fortran::semantics {

using namespace parser::literals;

class CanonicalizationOfOSS {
public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  CanonicalizationOfOSS(parser::Messages &messages) : messages_{messages} {}

  void Post(parser::Block &block) {
    for (auto it{block.begin()}; it != block.end(); ++it) {
      if (auto *ossCons{GetConstructIf<parser::OmpSsConstruct>(*it)}) {
        // OmpSsLoopConstruct
        if (auto *ossLoop{
                std::get_if<parser::OmpSsLoopConstruct>(&ossCons->u)}) {
          RewriteOmpSsLoopConstruct(*ossLoop, block, it);
        }
      } else if (auto *endDir{
                     GetConstructIf<parser::OSSEndLoopDirective>(*it)}) {
        // Unmatched OSSEndLoopDirective
        auto &dir{std::get<parser::OSSLoopDirective>(endDir->t)};
        messages_.Say(dir.source,
            "The %s directive must follow the DO loop associated with the "
            "loop construct"_err_en_US,
            parser::ToUpperCaseLetters(dir.source.ToString()));
      }
    } // Block list
  }

private:
  template <typename T> T *GetConstructIf(parser::ExecutionPartConstruct &x) {
    if (auto *y{std::get_if<parser::ExecutableConstruct>(&x.u)}) {
      if (auto *z{std::get_if<common::Indirection<T>>(&y->u)}) {
        return &z->value();
      }
    }
    return nullptr;
  }

  void RewriteOmpSsLoopConstruct(parser::OmpSsLoopConstruct &x,
      parser::Block &block, parser::Block::iterator it) {
    // Check the sequence of DoConstruct and OSSEndLoopDirective
    // in the same iteration
    //
    // Original:
    //   ExecutableConstruct -> OmpSsConstruct -> OmpSsLoopConstruct
    //     OSSBeginLoopDirective
    //   ExecutableConstruct -> DoConstruct
    //   ExecutableConstruct -> OSSEndLoopDirective (if available)
    //
    // After rewriting:
    //   ExecutableConstruct -> OmpSsConstruct -> OmpSsLoopConstruct
    //     OSSBeginLoopDirective
    //     DoConstruct
    //     OSSEndLoopDirective (if available)
    parser::Block::iterator nextIt;
    auto &beginDir{std::get<parser::OSSBeginLoopDirective>(x.t)};
    auto &dir{std::get<parser::OSSLoopDirective>(beginDir.t)};

    nextIt = it;
    if (++nextIt != block.end()) {
      if (auto *doCons{GetConstructIf<parser::DoConstruct>(*nextIt)}) {
        if (doCons->GetLoopControl()) {
          if (doCons->IsDoWhile()) {
            messages_.Say(dir.source,
                "DO WHILE after the %s directive is not supported"_err_en_US,
                parser::ToUpperCaseLetters(dir.source.ToString()));
          } else if (doCons->IsDoConcurrent()) {
            messages_.Say(dir.source,
                "DO CONCURRENT after the %s directive is not supported"_err_en_US,
                parser::ToUpperCaseLetters(dir.source.ToString()));
          }
          // move DoConstruct
          std::get<std::optional<parser::DoConstruct>>(x.t) =
              std::move(*doCons);
          nextIt = block.erase(nextIt);
          // try to match OSSEndLoopDirective
          if (nextIt != block.end()) {
            if (auto *endDir{
                    GetConstructIf<parser::OSSEndLoopDirective>(*nextIt)}) {
              std::get<std::optional<parser::OSSEndLoopDirective>>(x.t) =
                  std::move(*endDir);
              block.erase(nextIt);
            }
          }
        } else {
          messages_.Say(dir.source,
              "DO loop after the %s directive must have loop control"_err_en_US,
              parser::ToUpperCaseLetters(dir.source.ToString()));
        }
        return; // found do-loop
      }
    }
    messages_.Say(dir.source,
        "A DO loop must follow the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(dir.source.ToString()));
  }

  parser::Messages &messages_;
};

bool CanonicalizeOSS(parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfOSS oss{messages};
  Walk(program, oss);
  return !messages.AnyFatalError();
}
} // namespace Fortran::semantics
