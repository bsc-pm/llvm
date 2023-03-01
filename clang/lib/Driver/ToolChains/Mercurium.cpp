//===--- HLSL.cpp - HLSL ToolChain Implementations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mercurium.h"
#include "CommonArgs.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <string_view>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;
using namespace llvm;

namespace {
std::vector<std::string> splitParams(std::string_view input) {
  std::vector<std::string> res;
  std::string arg;
  auto pushArg = [&] {
    if (!arg.empty()) {
      res.push_back(std::move(arg));
      arg = std::string(); // reset
    }
  };
  unsigned pos = 0;
  bool inString = false;
  while (pos < input.size()) {
    switch (input[pos]) {
    case ' ':
      if (!inString) {
        pushArg();
        ++pos;
      } else {
        arg.push_back(input[pos++]);
      }
      break;
    case '"':
      inString = !inString;
      ++pos;
      break;
    case '\\':
      if (pos != input.size() - 1) {
        arg.push_back(input[++pos]);
      } else {
        ++pos;
      }
      ++pos;
      break;
    default:
      arg.push_back(input[pos++]);
      break;
    }
  }
  pushArg();
  return res;
}
} // namespace

void tools::mercurium::Mercurium::ConstructJob(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
    const char *LinkingOutput) const {
  FPGAMercuriumJobAction const &MJA = cast<FPGAMercuriumJobAction>(JA);
  auto &D = this->getToolChain().getDriver();
  std::string mercuriumTryPath = "fpgacxx";
  if (Arg *A = Args.getLastArg(options::OPT_fompss_fpga_mercurium); A) {
    mercuriumTryPath = A->getValue();
  }
  auto mercuriumExe = getToolChain().GetProgramPath(mercuriumTryPath.c_str());
  if (!llvm::sys::fs::can_execute(mercuriumExe)) {
    D.Diag(diag::err_drv_fpga_mercurium_could_not_locate) << mercuriumTryPath;
  }

  ArgStringList CmdArgs;
  for (auto &&A :
       Args.getAllArgValues(options::OPT_fompss_fpga_mercurium_flags)) {
    for (auto part : splitParams(A)) {
      CmdArgs.push_back(Args.MakeArgString(part));
    }
  }
  for (auto InputJob : MJA.getInputs()) {
    auto *WrappeGenInput = dyn_cast<FPGAWrapperGenJobAction>(InputJob);
    assert(WrappeGenInput && "The inputs to the Mercurium Job must be "
                             "FPGAWrapperWrapperGenJobAction");

    const auto *InFile = Args.MakeArgString(
        std::string(WrappeGenInput->getOutputDirPath()) + "/extracted." +
        (WrappeGenInput->getType() == types::TY_PP_C ? "c" : "cpp"));
    CmdArgs.push_back(InFile);
  }

  CmdArgs.push_back("-o");
  auto *mercuriumOutFile = D.CreateTempFile(C, "mercuriumFile", "");
  CmdArgs.push_back(mercuriumOutFile);
  CmdArgs.push_back("--fpga-link");
  const char *Exec = Args.MakeArgString(mercuriumExe);
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Inputs, Inputs));
}
