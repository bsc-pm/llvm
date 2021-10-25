// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

// In the past (35ff50076f43) I used 
// 
//    if (isOmpSsLoopDirective(DKind)) {
//      if (Tok.isNot(tok::kw_for)) {
//        // loop clauses start by 'for'
//        // Our strategy is to just drop the tokens if we do not guess
//        // a for is comming.
//        OSSLateParsedToks.clear();
//      }
//    }
//
// To avoid the following Sema infinite error. In this moment
// It seems to not be necessary, but I add a test to log this
// situation

// t1.c:3:33: error: use of undeclared identifier 'x'
//     #pragma oss taskloop shared(x)
//                                 ^
// t1.c:5:21: error: use of undeclared identifier 'taskloop'
//         #pragma oss taskloop
//                     ^
// t1.c:5:29: error: expected expression
//         #pragma oss taskloop
//                             ^
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// t1.c:5:29: error: expected expression
// fatal error: too many errors emitted, stopping now [-ferror-limit=]

int main() {
    int i;
    #pragma oss taskloop shared(x)
    while (i < 10) { // expected-error {{statement after '#pragma oss taskloop' must be a for loop}}
        #pragma oss taskloop
        for (i = 0; i < 10; i++) {
            i++;
        }
    }
}
