//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// MachineValueTypes.h file.  If you update anything here, you must update it
// there as well!
//
//===----------------------------------------------------------------------===//

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

def OtherVT: ValueType<0  ,  1>;   // "Other" value
def i1     : ValueType<1  ,  2>;   // One bit boolean value
def i2     : ValueType<2  ,  3>;   // 2-bit integer value
def i4     : ValueType<4  ,  4>;   // 4-bit integer value
def i8     : ValueType<8  ,  5>;   // 8-bit integer value
def i16    : ValueType<16 ,  6>;   // 16-bit integer value
def i32    : ValueType<32 ,  7>;   // 32-bit integer value
def i64    : ValueType<64 ,  8>;   // 64-bit integer value
def i128   : ValueType<128,  9>;   // 128-bit integer value
def f16    : ValueType<16 , 10>;   // 16-bit floating point value
def f32    : ValueType<32 , 11>;   // 32-bit floating point value
def f64    : ValueType<64 , 12>;   // 64-bit floating point value
def f80    : ValueType<80 , 13>;   // 80-bit floating point value
def f128   : ValueType<128, 14>;   // 128-bit floating point value
def ppcf128: ValueType<128, 15>;   // PPC 128-bit floating point value

def v1i1   : ValueType<1 ,  16>;   //   1 x i1 vector value
def v2i1   : ValueType<2 ,  17>;   //   2 x i1 vector value
def v4i1   : ValueType<4 ,  18>;   //   4 x i1 vector value
def v8i1   : ValueType<8 ,  19>;   //   8 x i1 vector value
def v16i1  : ValueType<16,  20>;   //  16 x i1 vector value
def v32i1  : ValueType<32 , 21>;   //  32 x i1 vector value
def v64i1  : ValueType<64 , 22>;   //  64 x i1 vector value
def v128i1 : ValueType<128, 23>;   // 128 x i1 vector value
def v512i1 : ValueType<512, 24>;   // 512 x i1 vector value
def v1024i1: ValueType<1024,25>;   //1024 x i1 vector value

def v1i8   : ValueType<8,   26>;   //  1 x i8  vector value
def v2i8   : ValueType<16 , 27>;   //  2 x i8  vector value
def v4i8   : ValueType<32 , 28>;   //  4 x i8  vector value
def v8i8   : ValueType<64 , 29>;   //  8 x i8  vector value
def v16i8  : ValueType<128, 30>;   // 16 x i8  vector value
def v32i8  : ValueType<256, 31>;   // 32 x i8  vector value
def v64i8  : ValueType<512, 32>;   // 64 x i8  vector value
def v128i8 : ValueType<1024,33>;   //128 x i8  vector value
def v256i8 : ValueType<2048,34>;   //256 x i8  vector value

def v1i16  : ValueType<16 , 35>;   //  1 x i16 vector value
def v2i16  : ValueType<32 , 36>;   //  2 x i16 vector value
def v3i16  : ValueType<48 , 37>;   //  3 x i16 vector value
def v4i16  : ValueType<64 , 38>;   //  4 x i16 vector value
def v8i16  : ValueType<128, 39>;   //  8 x i16 vector value
def v16i16 : ValueType<256, 40>;   // 16 x i16 vector value
def v32i16 : ValueType<512, 41>;   // 32 x i16 vector value
def v64i16 : ValueType<1024,42>;   // 64 x i16 vector value
def v128i16: ValueType<2048,43>;   //128 x i16 vector value

def v1i32    : ValueType<32 , 44>;   //  1 x i32 vector value
def v2i32    : ValueType<64 , 45>;   //  2 x i32 vector value
def v3i32    : ValueType<96 , 46>;   //  3 x i32 vector value
def v4i32    : ValueType<128, 47>;   //  4 x i32 vector value
def v5i32    : ValueType<160, 48>;   //  5 x i32 vector value
def v8i32    : ValueType<256, 49>;   //  8 x i32 vector value
def v16i32   : ValueType<512, 50>;   // 16 x i32 vector value
def v32i32   : ValueType<1024,51>;   // 32 x i32 vector value
def v64i32   : ValueType<2048,52>;   // 64 x i32 vector value
def v128i32  : ValueType<4096,53>;   // 128 x i32 vector value
def v256i32  : ValueType<8182,54>;   // 256 x i32 vector value
def v512i32  : ValueType<16384,55>;  // 512 x i32 vector value
def v1024i32 : ValueType<32768,56>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536,57>;  // 2048 x i32 vector value

def v1i64  : ValueType<64 , 58>;   //  1 x i64 vector value
def v2i64  : ValueType<128, 59>;   //  2 x i64 vector value
def v4i64  : ValueType<256, 60>;   //  4 x i64 vector value
def v8i64  : ValueType<512, 61>;   //  8 x i64 vector value
def v16i64 : ValueType<1024,62>;   // 16 x i64 vector value
def v32i64 : ValueType<2048,63>;   // 32 x i64 vector value

def v1i128 : ValueType<128, 64>;   //  1 x i128 vector value

def nxv1i1  : ValueType<1,   65>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,   66>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,   67>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,   68>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16,  69>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32,  70>;  // n x 32 x i1  vector value

def nxv1i8  : ValueType<8,   71>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  72>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  73>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  74>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 75>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 76>;  // n x 32 x i8  vector value

def nxv1i16 : ValueType<16,  77>;  // n x  1 x i16 vector value
def nxv2i16 : ValueType<32,  78>;  // n x  2 x i16 vector value
def nxv4i16 : ValueType<64,  79>;  // n x  4 x i16 vector value
def nxv8i16 : ValueType<128, 80>;  // n x  8 x i16 vector value
def nxv16i16: ValueType<256, 81>;  // n x 16 x i16 vector value
def nxv32i16: ValueType<512, 82>;  // n x 32 x i16 vector value

def nxv1i32 : ValueType<32,  83>;  // n x  1 x i32 vector value
def nxv2i32 : ValueType<64,  84>;  // n x  2 x i32 vector value
def nxv4i32 : ValueType<128, 85>;  // n x  4 x i32 vector value
def nxv8i32 : ValueType<256, 86>;  // n x  8 x i32 vector value
def nxv16i32: ValueType<512, 87>;  // n x 16 x i32 vector value
def nxv32i32: ValueType<1024,88>;  // n x 32 x i32 vector value

def nxv1i64 : ValueType<64,  89>;  // n x  1 x i64 vector value
def nxv2i64 : ValueType<128, 90>;  // n x  2 x i64 vector value
def nxv4i64 : ValueType<256, 91>;  // n x  4 x i64 vector value
def nxv8i64 : ValueType<512, 92>;  // n x  8 x i64 vector value
def nxv16i64: ValueType<1024,93>;  // n x 16 x i64 vector value
def nxv32i64: ValueType<2048,94>;  // n x 32 x i64 vector value

// EPIVR Masks
def nxv16i4 : ValueType<64, 95>;   // n x 16 x i4 vector value
def nxv32i2 : ValueType<64, 96>;   // n x 32 x i2 vector value
def nxv64i1 : ValueType<64, 97>;   // n x 64 x i1 vector value

def v2f16    : ValueType<32 , 98>;    //    2 x f16 vector value
def v3f16    : ValueType<48 , 99>;    //    3 x f16 vector value
def v4f16    : ValueType<64 , 100>;    //    4 x f16 vector value
def v8f16    : ValueType<128, 101>;    //    8 x f16 vector value
def v16f16   : ValueType<256, 102>;   //    8 x f16 vector value
def v32f16   : ValueType<512, 103>;   //    8 x f16 vector value
def v1f32    : ValueType<32 , 104>;   //    1 x f32 vector value
def v2f32    : ValueType<64 , 105>;   //    2 x f32 vector value
def v3f32    : ValueType<96 , 106>;   //    3 x f32 vector value
def v4f32    : ValueType<128, 107>;   //    4 x f32 vector value
def v5f32    : ValueType<160, 108>;   //    5 x f32 vector value
def v8f32    : ValueType<256, 109>;   //    8 x f32 vector value
def v16f32   : ValueType<512,  110>;  //   16 x f32 vector value
def v32f32   : ValueType<1024, 111>;  //   32 x f32 vector value
def v64f32   : ValueType<2048, 112>;  //   64 x f32 vector value
def v128f32  : ValueType<4096, 113>;  //  128 x f32 vector value
def v256f32  : ValueType<8182, 114>;  //  256 x f32 vector value
def v512f32  : ValueType<16384, 115>; //  512 x f32 vector value
def v1024f32 : ValueType<32768, 116>; // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 117>; // 2048 x f32 vector value
def v1f64    : ValueType<64, 118>;    //    1 x f64 vector value
def v2f64    : ValueType<128, 119>;   //    2 x f64 vector value
def v4f64    : ValueType<256, 120>;   //    4 x f64 vector value
def v8f64    : ValueType<512, 121>;   //    8 x f64 vector value

def nxv2f16  : ValueType<32 , 122>; // n x  2 x f16 vector value
def nxv4f16  : ValueType<64 , 123>; // n x  4 x f16 vector value
def nxv8f16  : ValueType<128, 124>; // n x  8 x f16 vector value
def nxv1f32  : ValueType<32 , 125>; // n x  1 x f32 vector value
def nxv2f32  : ValueType<64 , 126>; // n x  2 x f32 vector value
def nxv4f32  : ValueType<128, 127>; // n x  4 x f32 vector value
def nxv8f32  : ValueType<256, 128>; // n x  8 x f32 vector value
def nxv16f32 : ValueType<512, 129>; // n x 16 x f32 vector value
def nxv1f64  : ValueType<64,  130>; // n x  1 x f64 vector value
def nxv2f64  : ValueType<128, 131>; // n x  2 x f64 vector value
def nxv4f64  : ValueType<256, 132>; // n x  4 x f64 vector value
def nxv8f64  : ValueType<512, 133>; // n x  8 x f64 vector value

def x86mmx : ValueType<64 , 134>;   // X86 MMX value
def FlagVT : ValueType<0  , 135>;   // Pre-RA sched glue
def isVoid : ValueType<0  , 136>;   // Produces no value
def untyped: ValueType<8  , 137>;   // Produces an untyped value
def exnref: ValueType<0, 138>;      // WebAssembly's exnref type
def token  : ValueType<0  , 248>;   // TokenTy
def MetadataVT: ValueType<0, 249>;  // Metadata

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny   : ValueType<0, 250>;

// Pseudo valuetype to represent "vector of any size"
def vAny   : ValueType<0  , 251>;

// Pseudo valuetype to represent "float of any format"
def fAny   : ValueType<0  , 252>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny   : ValueType<0  , 253>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR   : ValueType<0  , 254>;

// Pseudo valuetype to represent "any type of any size".
def Any    : ValueType<0  , 255>;

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the ponter.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
