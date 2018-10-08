//=- RISCVInstrFormatsV.td - RISCV EPI Instruction Formats --*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file describes the RISC-V EPI extension instruction formats.
//
//===----------------------------------------------------------------------===//

class RVInstEPIOPV<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;
  bits<1> vm;
  bits<3> vops;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = vops;
  let Inst{11-7} = rd;
  let Opcode = OPC_V.Value;
}

class RVInstEPIOPV_IVV<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInstEPIOPV<funct6, outs, ins, opcodestr, argstr>
{
   let vops = 0b000;
}

class RVInstEPIOPV_FVV<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInstEPIOPV<funct6, outs, ins, opcodestr, argstr>
{
   let vops = 0b001;
}

class RVInstEPIOPV_MVV<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInstEPIOPV<funct6, outs, ins, opcodestr, argstr>
{
   let vops = 0b010;
}

// This is the same as RVInstEPIOPV, it is just convenient to
// rename the immediate field. vops is hardcoded.
class RVInstEPIOPV_IVI<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther>
{
  bits<5> rs2;
  bits<5> imm5;
  bits<5> rd;
  bits<1> vm;
  bits<3> vops;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = imm5;
  let Inst{14-12} = 0b011;
  let Inst{11-7} = rd;
  let Opcode = OPC_V.Value;
}

class RVInstEPIOPV_IVX<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInstEPIOPV<funct6, outs, ins, opcodestr, argstr>
{
   let vops = 0b100;
}

class RVInstEPIOPV_FVF<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInstEPIOPV<funct6, outs, ins, opcodestr, argstr>
{
   let vops = 0b101;
}

class RVInstEPIOPV_MVX<bits<6> funct6,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInstEPIOPV<funct6, outs, ins, opcodestr, argstr>
{
   let vops = 0b110;
}

class RVInstEPIOPVAMO<bits<5> amoop,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<1> wd;
  bits<1> vm;
  bits<5> rs2;
  bits<5> rs1;
  bits<3> width;
  bits<5> rd;

  let Inst{31-27} = amoop;
  let Inst{26} = wd;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = rd;
  let Opcode = OPC_AMO.Value;
}

class RVInstMem<bits<3> mop, RISCVOpcode opcode,
        dag outs, dag ins, string opcodestr, string argstr>
        : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther>
{
  bits<3> nf;
  bits<1> vm;
  bits<5> rs2; // lumop / sumop / rs2 / vs2
  bits<5> rs1;
  bits<3> width;
  bits<5> rd; // rs3

  let Inst{31-29} = nf;
  let Inst{28-26} = mop;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = rd;
  let Opcode = opcode.Value;
}

class RVInstLoad<bits<3> mop, dag outs, dag ins, string opcodestr,
        string argstr> : RVInstMem<mop, OPC_LOAD_FP, outs, ins,
                               opcodestr, argstr>;


class RVInstStore<bits<3> mop, dag outs, dag ins, string opcodestr,
        string argstr> : RVInstMem<mop, OPC_STORE_FP, outs, ins,
                               opcodestr, argstr>;
