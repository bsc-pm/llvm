//===--- Ait.cpp - Ait ToolChain Implementation -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ait.h"
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
#include <fstream>
#include <optional>
#include <sstream>
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

std::string MergeJsonParts(const FPGAAitJobAction &AitJA) {
  std::string outJsonFileStr("[\n");
  llvm::raw_string_ostream outJsonFile(outJsonFileStr);
  for (auto *InputJob : AitJA.getInputs()) {
    auto *WrappeGenInput = dyn_cast<FPGAWrapperGenJobAction>(InputJob);
    assert(WrappeGenInput && "The inputs to the Mercurium Job must be "
                             "FPGAWrapperWrapperGenJobAction");
    std::ifstream t(std::string(WrappeGenInput->getOutputDirPath()) +
                    "/extracted.json.part");
    assert(t.is_open());
    t.seekg(0, std::ios::end);
    size_t size = t.tellg();
    std::string buffer(size, '\0');
    t.seekg(0);
    t.read(&buffer[0], size);
    outJsonFile << buffer;
  }
  outJsonFileStr.pop_back(); // remove \n
  outJsonFileStr.pop_back(); // remove ,
  outJsonFileStr.append("\n]\n");
  return outJsonFileStr;
}

/// Command that just runs an internal function.
class FuncCommand : public Command {
  std::function<void()> Exec;
  std::optional<Command> OtherCommand;

public:
  FuncCommand(std::function<void()> &&Exec, const Action &Source,
              const Tool &Creator, ArrayRef<InputInfo> Inputs,
              ArrayRef<InputInfo> Outputs = std::nullopt,
              std::optional<Command> &&OtherCommand = std::nullopt)
      : Command(Source, Creator, ResponseFileSupport::None(), nullptr,
                ArgStringList{}, Inputs, Outputs),
        Exec(std::move(Exec)), OtherCommand(std::move(OtherCommand)) {}
  // FIXME: This really shouldn't be copyable, but is currently copied in some
  // error handling in Driver::generateCompilationDiagnostics.
  FuncCommand(const FuncCommand &) = default;
  ~FuncCommand() override = default;

  void Print(llvm::raw_ostream &OS, const char *Terminator, bool Quote,
             CrashReportInfo *CrashInfo = nullptr) const override {
    OS << "FuncCommand: <function pointer>";
  }

  int Execute(ArrayRef<std::optional<StringRef>> Redirects, std::string *ErrMsg,
              bool *ExecutionFailed) const override {
    Exec();
    if (OtherCommand.has_value()) {
      return OtherCommand->Execute(Redirects, ErrMsg, ExecutionFailed);
    }
    return 0;
  }
};

} // namespace

void tools::ait::Ait::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  FPGAAitJobAction const &AitJA = cast<FPGAAitJobAction>(JA);
  auto &D = this->getToolChain().getDriver();
  std::string AitTryPath = "ait";
  if (Arg *A = Args.getLastArg(options::OPT_fompss_fpga_ait); A) {
    AitTryPath = A->getValue();
  }
  if (!llvm::sys::fs::can_execute(AitTryPath)) {
    auto AitExe = getToolChain().GetProgramPath(AitTryPath.c_str());
    if (!llvm::sys::fs::can_execute(AitExe)) {
      D.Diag(diag::err_drv_fpga_ait_could_not_locate) << AitTryPath;
    }
    AitTryPath = AitExe;
  }

  ArgStringList CmdArgs;
  for (auto &&A : Args.getAllArgValues(options::OPT_fompss_fpga_ait_flags)) {
    for (auto part : splitParams(A)) {
      CmdArgs.push_back(Args.MakeArgString(part));
    }
  }
  CmdArgs.push_back(Args.MakeArgString("--wrapper_version"));
  CmdArgs.push_back(Args.MakeArgString("14"));

  auto *JsonFilePath = C.addTempFile("ait_extracted.json");
  auto command = [JsonFilePath, AitJA]() -> void {
    std::ofstream outputJsonFile(JsonFilePath);
    outputJsonFile << MergeJsonParts(AitJA);
  };
  const char *Exec = Args.MakeArgString(AitTryPath);
  C.addCommand(std::unique_ptr<Command>(new FuncCommand(
      std::move(command), JA, *this, Inputs, Inputs,
      std::make_optional<Command>(JA, *this, ResponseFileSupport::None(), Exec,
                                  CmdArgs, Inputs, Inputs))));
}
