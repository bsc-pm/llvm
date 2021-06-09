// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 -o - %s

int v[10];
int main() {
    #pragma oss task in( { v[i+j], i=i;i:i, j=i:1 } ) // expected-error {{iterator variable of a multidependence cannot be used in its own lower-bound expression}} expected-error {{iterator variable of a multidependence cannot be used in its own size expression}} expected-error {{iterator variable of a multidependence cannot be used in its own step expression}}
    {}
    #pragma oss task in( { v[0], i=0;v[i] } ) // expected-error {{iterator variable of a multidependence cannot be used in its own size expression}}
    {}
}

