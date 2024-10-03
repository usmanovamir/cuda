#include "secp256k1.h"
#include "def.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __host__ u32 sub(u32* r, const u32* a, const u32* b) {
    u32 c = 0; // carry/borrow
    for (u32 i = 0; i < 8; i++) {
        const u32 diff = a[i] - b[i] - c;
        if (diff != a[i]) c = (diff > a[i]);
        r[i] = diff;
    }
    return c;
}

__device__ __host__ u32 add(u32* r, const u32* a, const u32* b) {
    u32 c = 0; // carry/borrow
    for (u32 i = 0; i < 8; i++) {
        const u32 t = a[i] + b[i] + c;
        if (t != a[i]) c = (t < a[i]);
        r[i] = t;
    }
    return c;
}

__device__ void sub_mod(u32* r, const u32* a, const u32* b) {
    const u32 c = sub(r, a, b); // carry
    if (c) {
        u32 t[8];
        t[0] = SECP256K1_P0;
        t[1] = SECP256K1_P1;
        t[2] = SECP256K1_P2;
        t[3] = SECP256K1_P3;
        t[4] = SECP256K1_P4;
        t[5] = SECP256K1_P5;
        t[6] = SECP256K1_P6;
        t[7] = SECP256K1_P7;
        add(r, r, t);
    }
}

__device__ void add_mod(u32* r, const u32* a, const u32* b) {
    const u32 c = add(r, a, b);
    u32 t[8];
    t[0] = SECP256K1_P0;
    t[1] = SECP256K1_P1;
    t[2] = SECP256K1_P2;
    t[3] = SECP256K1_P3;
    t[4] = SECP256K1_P4;
    t[5] = SECP256K1_P5;
    t[6] = SECP256K1_P6;
    t[7] = SECP256K1_P7;
    if (c == 0) {
        for (int i = 7; i >= 0; i--) {
            if (r[i] < t[i]) return;
            if (r[i] > t[i]) break;
        }
    }
    sub(r, r, t);
}

__device__ __host__ void mul_mod(u32* r, const u32* a, const u32* b) {
    u32 t[16] = { 0 };
    u32 t0 = 0;
    u32 t1 = 0;
    u32 c = 0;
    for (u32 i = 0; i < 8; i++) {
        for (u32 j = 0; j <= i; j++) {
            u64 p = ((u64)a[j]) * b[i - j];
            u64 d = ((u64)t1) << 32 | t0;
            d += p;
            t0 = (u32)d;
            t1 = d >> 32;
            c += d < p; // carry
        }
        t[i] = t0;
        t0 = t1;
        t1 = c;
        c = 0;
    }
    for (u32 i = 8; i < 15; i++) {
        for (u32 j = i - 7; j < 8; j++) {
            u64 p = ((u64)a[j]) * b[i - j];
            u64 d = ((u64)t1) << 32 | t0;
            d += p;
            t0 = (u32)d;
            t1 = d >> 32;
            c += d < p;
        }
        t[i] = t0;
        t0 = t1;
        t1 = c;
        c = 0;
    }
    t[15] = t0;
    u32 tmp[16] = { 0 };
    for (u32 i = 0, j = 8; i < 8; i++, j++) {
        u64 p = ((u64)0x03d1) * t[j] + c;
        tmp[i] = (u32)p;
        c = p >> 32;
    }
    tmp[8] = c;
    c = add(tmp + 1, tmp + 1, t + 8); // modifies tmp[1]...tmp[8]
    tmp[9] = c;
    c = add(r, t, tmp);
    u32 c2 = 0;
    for (u32 i = 0, j = 8; i < 8; i++, j++) {
        u64 p = ((u64)0x3d1) * tmp[j] + c2;
        t[i] = (u32)p;
        c2 = p >> 32;
    }
    t[8] = c2;
    c2 = add(t + 1, t + 1, tmp + 8); // modifies t[1]...t[8]
    t[9] = c2;
    c2 = add(r, r, t);
    c += c2;
    t[0] = SECP256K1_P0;
    t[1] = SECP256K1_P1;
    t[2] = SECP256K1_P2;
    t[3] = SECP256K1_P3;
    t[4] = SECP256K1_P4;
    t[5] = SECP256K1_P5;
    t[6] = SECP256K1_P6;
    t[7] = SECP256K1_P7;
    for (u32 i = c; i > 0; i--) {
        sub(r, r, t);
    }
    for (int i = 7; i >= 0; i--) {
        if (r[i] < t[i]) break;
        if (r[i] > t[i]) {
            sub(r, r, t);
            break;
        }
    }
}

__device__ void inv_mod(u32* a) {
    u32 t0[8];
    t0[0] = a[0];
    t0[1] = a[1];
    t0[2] = a[2];
    t0[3] = a[3];
    t0[4] = a[4];
    t0[5] = a[5];
    t0[6] = a[6];
    t0[7] = a[7];
    u32 p[8];
    p[0] = SECP256K1_P0;
    p[1] = SECP256K1_P1;
    p[2] = SECP256K1_P2;
    p[3] = SECP256K1_P3;
    p[4] = SECP256K1_P4;
    p[5] = SECP256K1_P5;
    p[6] = SECP256K1_P6;
    p[7] = SECP256K1_P7;
    u32 t1[8];
    t1[0] = SECP256K1_P0;
    t1[1] = SECP256K1_P1;
    t1[2] = SECP256K1_P2;
    t1[3] = SECP256K1_P3;
    t1[4] = SECP256K1_P4;
    t1[5] = SECP256K1_P5;
    t1[6] = SECP256K1_P6;
    t1[7] = SECP256K1_P7;
    u32 t2[8] = { 0 };
    t2[0] = 0x00000001;
    u32 t3[8] = { 0 };
    u32 b = (t0[0] != t1[0])
        | (t0[1] != t1[1])
        | (t0[2] != t1[2])
        | (t0[3] != t1[3])
        | (t0[4] != t1[4])
        | (t0[5] != t1[5])
        | (t0[6] != t1[6])
        | (t0[7] != t1[7]);
    while (b) {
        if ((t0[0] & 1) == 0) {
            t0[0] = t0[0] >> 1 | t0[1] << 31;
            t0[1] = t0[1] >> 1 | t0[2] << 31;
            t0[2] = t0[2] >> 1 | t0[3] << 31;
            t0[3] = t0[3] >> 1 | t0[4] << 31;
            t0[4] = t0[4] >> 1 | t0[5] << 31;
            t0[5] = t0[5] >> 1 | t0[6] << 31;
            t0[6] = t0[6] >> 1 | t0[7] << 31;
            t0[7] = t0[7] >> 1;
            u32 c = 0;
            if (t2[0] & 1) c = add(t2, t2, p);
            t2[0] = t2[0] >> 1 | t2[1] << 31;
            t2[1] = t2[1] >> 1 | t2[2] << 31;
            t2[2] = t2[2] >> 1 | t2[3] << 31;
            t2[3] = t2[3] >> 1 | t2[4] << 31;
            t2[4] = t2[4] >> 1 | t2[5] << 31;
            t2[5] = t2[5] >> 1 | t2[6] << 31;
            t2[6] = t2[6] >> 1 | t2[7] << 31;
            t2[7] = t2[7] >> 1 | c << 31;
        } else if ((t1[0] & 1) == 0) {
            t1[0] = t1[0] >> 1 | t1[1] << 31;
            t1[1] = t1[1] >> 1 | t1[2] << 31;
            t1[2] = t1[2] >> 1 | t1[3] << 31;
            t1[3] = t1[3] >> 1 | t1[4] << 31;
            t1[4] = t1[4] >> 1 | t1[5] << 31;
            t1[5] = t1[5] >> 1 | t1[6] << 31;
            t1[6] = t1[6] >> 1 | t1[7] << 31;
            t1[7] = t1[7] >> 1;
            u32 c = 0;
            if (t3[0] & 1) c = add(t3, t3, p);
            t3[0] = t3[0] >> 1 | t3[1] << 31;
            t3[1] = t3[1] >> 1 | t3[2] << 31;
            t3[2] = t3[2] >> 1 | t3[3] << 31;
            t3[3] = t3[3] >> 1 | t3[4] << 31;
            t3[4] = t3[4] >> 1 | t3[5] << 31;
            t3[5] = t3[5] >> 1 | t3[6] << 31;
            t3[6] = t3[6] >> 1 | t3[7] << 31;
            t3[7] = t3[7] >> 1 | c << 31;
        } else {
            u32 gt = 0;
            for (int i = 7; i >= 0; i--) {
                if (t0[i] > t1[i]) {
                    gt = 1;
                    break;
                }
                if (t0[i] < t1[i]) break;
            }
            if (gt) {
                sub(t0, t0, t1);
                t0[0] = t0[0] >> 1 | t0[1] << 31;
                t0[1] = t0[1] >> 1 | t0[2] << 31;
                t0[2] = t0[2] >> 1 | t0[3] << 31;
                t0[3] = t0[3] >> 1 | t0[4] << 31;
                t0[4] = t0[4] >> 1 | t0[5] << 31;
                t0[5] = t0[5] >> 1 | t0[6] << 31;
                t0[6] = t0[6] >> 1 | t0[7] << 31;
                t0[7] = t0[7] >> 1;
                u32 lt = 0;
                for (int i = 7; i >= 0; i--) {
                    if (t2[i] < t3[i]) {
                        lt = 1;
                        break;
                    }
                    if (t2[i] > t3[i]) break;
                }
                if (lt) add(t2, t2, p);
                sub(t2, t2, t3);
                u32 c = 0;
                if (t2[0] & 1) c = add(t2, t2, p);
                t2[0] = t2[0] >> 1 | t2[1] << 31;
                t2[1] = t2[1] >> 1 | t2[2] << 31;
                t2[2] = t2[2] >> 1 | t2[3] << 31;
                t2[3] = t2[3] >> 1 | t2[4] << 31;
                t2[4] = t2[4] >> 1 | t2[5] << 31;
                t2[5] = t2[5] >> 1 | t2[6] << 31;
                t2[6] = t2[6] >> 1 | t2[7] << 31;
                t2[7] = t2[7] >> 1 | c << 31;
            } else {
                sub(t1, t1, t0);
                t1[0] = t1[0] >> 1 | t1[1] << 31;
                t1[1] = t1[1] >> 1 | t1[2] << 31;
                t1[2] = t1[2] >> 1 | t1[3] << 31;
                t1[3] = t1[3] >> 1 | t1[4] << 31;
                t1[4] = t1[4] >> 1 | t1[5] << 31;
                t1[5] = t1[5] >> 1 | t1[6] << 31;
                t1[6] = t1[6] >> 1 | t1[7] << 31;
                t1[7] = t1[7] >> 1;
                u32 lt = 0;
                for (int i = 7; i >= 0; i--) {
                    if (t3[i] < t2[i]) {
                        lt = 1;
                        break;
                    }
                    if (t3[i] > t2[i]) break;
                }
                if (lt) add(t3, t3, p);
                sub(t3, t3, t2);
                u32 c = 0;
                if (t3[0] & 1) c = add(t3, t3, p);
                t3[0] = t3[0] >> 1 | t3[1] << 31;
                t3[1] = t3[1] >> 1 | t3[2] << 31;
                t3[2] = t3[2] >> 1 | t3[3] << 31;
                t3[3] = t3[3] >> 1 | t3[4] << 31;
                t3[4] = t3[4] >> 1 | t3[5] << 31;
                t3[5] = t3[5] >> 1 | t3[6] << 31;
                t3[6] = t3[6] >> 1 | t3[7] << 31;
                t3[7] = t3[7] >> 1 | c << 31;
            }
        }
        b = (t0[0] != t1[0])
            | (t0[1] != t1[1])
            | (t0[2] != t1[2])
            | (t0[3] != t1[3])
            | (t0[4] != t1[4])
            | (t0[5] != t1[5])
            | (t0[6] != t1[6])
            | (t0[7] != t1[7]);
    }
    a[0] = t2[0];
    a[1] = t2[1];
    a[2] = t2[2];
    a[3] = t2[3];
    a[4] = t2[4];
    a[5] = t2[5];
    a[6] = t2[6];
    a[7] = t2[7];
}

__device__ void point_double(u32* x, u32* y, u32* z) {
    u32 t1[8];
    t1[0] = x[0];
    t1[1] = x[1];
    t1[2] = x[2];
    t1[3] = x[3];
    t1[4] = x[4];
    t1[5] = x[5];
    t1[6] = x[6];
    t1[7] = x[7];
    u32 t2[8];
    t2[0] = y[0];
    t2[1] = y[1];
    t2[2] = y[2];
    t2[3] = y[3];
    t2[4] = y[4];
    t2[5] = y[5];
    t2[6] = y[6];
    t2[7] = y[7];
    u32 t3[8];
    t3[0] = z[0];
    t3[1] = z[1];
    t3[2] = z[2];
    t3[3] = z[3];
    t3[4] = z[4];
    t3[5] = z[5];
    t3[6] = z[6];
    t3[7] = z[7];
    u32 t4[8];
    u32 t5[8];
    u32 t6[8];
    mul_mod(t4, t1, t1); // t4 = x^2
    mul_mod(t5, t2, t2); // t5 = y^2
    mul_mod(t1, t1, t5); // t1 = x*y^2
    mul_mod(t5, t5, t5); // t5 = t5^2 = y^4
    mul_mod(t3, t2, t3); // t3 = x * z
    add_mod(t2, t4, t4); // t2 = 2 * t4 = 2 * x^2
    add_mod(t4, t4, t2); // t4 = 3 * t4 = 3 * x^2
    u32 c = 0;
    if (t4[0] & 1) {
        u32 t[8];
        t[0] = SECP256K1_P0;
        t[1] = SECP256K1_P1;
        t[2] = SECP256K1_P2;
        t[3] = SECP256K1_P3;
        t[4] = SECP256K1_P4;
        t[5] = SECP256K1_P5;
        t[6] = SECP256K1_P6;
        t[7] = SECP256K1_P7;
        c = add(t4, t4, t); // t4 + SECP256K1_P
    }
    t4[0] = t4[0] >> 1 | t4[1] << 31;
    t4[1] = t4[1] >> 1 | t4[2] << 31;
    t4[2] = t4[2] >> 1 | t4[3] << 31;
    t4[3] = t4[3] >> 1 | t4[4] << 31;
    t4[4] = t4[4] >> 1 | t4[5] << 31;
    t4[5] = t4[5] >> 1 | t4[6] << 31;
    t4[6] = t4[6] >> 1 | t4[7] << 31;
    t4[7] = t4[7] >> 1 | c << 31;
    mul_mod(t6, t4, t4); // t6 = t4^2 = (3/2 * x^2)^2
    add_mod(t2, t1, t1); // t2 = 2 * t1
    sub_mod(t6, t6, t2); // t6 = t6 - t2
    sub_mod(t1, t1, t6); // t1 = t1 - t6
    mul_mod(t4, t4, t1); // t4 = t4 * t1
    sub_mod(t1, t4, t5); // t1 = t4 - t5
    x[0] = t6[0];
    x[1] = t6[1];
    x[2] = t6[2];
    x[3] = t6[3];
    x[4] = t6[4];
    x[5] = t6[5];
    x[6] = t6[6];
    x[7] = t6[7];
    y[0] = t1[0];
    y[1] = t1[1];
    y[2] = t1[2];
    y[3] = t1[3];
    y[4] = t1[4];
    y[5] = t1[5];
    y[6] = t1[6];
    y[7] = t1[7];
    z[0] = t3[0];
    z[1] = t3[1];
    z[2] = t3[2];
    z[3] = t3[3];
    z[4] = t3[4];
    z[5] = t3[5];
    z[6] = t3[6];
    z[7] = t3[7];
}

__device__ void point_add(u32* x1, u32* y1, u32* z1, const u32* x2, const u32* y2) {
    u32 t1[8];
    t1[0] = x1[0];
    t1[1] = x1[1];
    t1[2] = x1[2];
    t1[3] = x1[3];
    t1[4] = x1[4];
    t1[5] = x1[5];
    t1[6] = x1[6];
    t1[7] = x1[7];
    u32 t2[8];
    t2[0] = y1[0];
    t2[1] = y1[1];
    t2[2] = y1[2];
    t2[3] = y1[3];
    t2[4] = y1[4];
    t2[5] = y1[5];
    t2[6] = y1[6];
    t2[7] = y1[7];
    u32 t3[8];
    t3[0] = z1[0];
    t3[1] = z1[1];
    t3[2] = z1[2];
    t3[3] = z1[3];
    t3[4] = z1[4];
    t3[5] = z1[5];
    t3[6] = z1[6];
    t3[7] = z1[7];
    u32 t4[8];
    t4[0] = x2[0];
    t4[1] = x2[1];
    t4[2] = x2[2];
    t4[3] = x2[3];
    t4[4] = x2[4];
    t4[5] = x2[5];
    t4[6] = x2[6];
    t4[7] = x2[7];
    u32 t5[8];
    t5[0] = y2[0];
    t5[1] = y2[1];
    t5[2] = y2[2];
    t5[3] = y2[3];
    t5[4] = y2[4];
    t5[5] = y2[5];
    t5[6] = y2[6];
    t5[7] = y2[7];
    u32 t6[8];
    u32 t7[8];
    u32 t8[8];
    u32 t9[8];
    mul_mod(t6, t3, t3); // t6 = t3^2
    mul_mod(t7, t6, t3); // t7 = t6*t3
    mul_mod(t6, t6, t4); // t6 = t6*t4
    mul_mod(t7, t7, t5); // t7 = t7*t5
    sub_mod(t6, t6, t1); // t6 = t6-t1
    sub_mod(t7, t7, t2); // t7 = t7-t2
    mul_mod(t8, t3, t6); // t8 = t3*t6
    mul_mod(t4, t6, t6); // t4 = t6^2
    mul_mod(t9, t4, t6); // t9 = t4*t6
    mul_mod(t4, t4, t1); // t4 = t4*t1
    t6[7] = t4[7] << 1 | t4[6] >> 31;
    t6[6] = t4[6] << 1 | t4[5] >> 31;
    t6[5] = t4[5] << 1 | t4[4] >> 31;
    t6[4] = t4[4] << 1 | t4[3] >> 31;
    t6[3] = t4[3] << 1 | t4[2] >> 31;
    t6[2] = t4[2] << 1 | t4[1] >> 31;
    t6[1] = t4[1] << 1 | t4[0] >> 31;
    t6[0] = t4[0] << 1;
    if (t4[7] & 0x80000000) {
        u32 a[8] = { 0 };
        a[1] = 1;
        a[0] = 0x000003d1; // omega (see: mul_mod ())
        add(t6, t6, a);
    }
    mul_mod(t5, t7, t7); // t5 = t7*t7
    sub_mod(t5, t5, t6); // t5 = t5-t6
    sub_mod(t5, t5, t9); // t5 = t5-t9
    sub_mod(t4, t4, t5); // t4 = t4-t5
    mul_mod(t4, t4, t7); // t4 = t4*t7
    mul_mod(t9, t9, t2); // t9 = t9*t2
    sub_mod(t9, t4, t9); // t9 = t4-t9
    x1[0] = t5[0];
    x1[1] = t5[1];
    x1[2] = t5[2];
    x1[3] = t5[3];
    x1[4] = t5[4];
    x1[5] = t5[5];
    x1[6] = t5[6];
    x1[7] = t5[7];
    y1[0] = t9[0];
    y1[1] = t9[1];
    y1[2] = t9[2];
    y1[3] = t9[3];
    y1[4] = t9[4];
    y1[5] = t9[5];
    y1[6] = t9[6];
    y1[7] = t9[7];
    z1[0] = t8[0];
    z1[1] = t8[1];
    z1[2] = t8[2];
    z1[3] = t8[3];
    z1[4] = t8[4];
    z1[5] = t8[5];
    z1[6] = t8[6];
    z1[7] = t8[7];
}

__device__ int convert_to_window_naf(u32* naf, const u32* k) {
    int loop_start = 0;
    u32 n[9];
    n[0] = 0; 
    n[1] = k[7];
    n[2] = k[6];
    n[3] = k[5];
    n[4] = k[4];
    n[5] = k[3];
    n[6] = k[2];
    n[7] = k[1];
    n[8] = k[0];
    for (int i = 0; i <= 256; i++) {
        if (n[8] & 1) {
            int diff = n[8] & 0x0f; 
            int val = diff;
            if (diff >= 0x08) {
                diff -= 0x10;
                val = 0x11 - val;
            }
            naf[i >> 3] |= val << ((i & 7) << 2);
            u32 t = n[8]; 
            n[8] -= diff;
            u32 k = 8;
            if (diff > 0) {
                while (n[k] > t) {
                    if (k == 0) break;
                    k--;
                    t = n[k];
                    n[k]--;
                }
            }
            else {
                while (t > n[k]) {
                    if (k == 0) break;
                    k--;
                    t = n[k];
                    n[k]++;
                }
            }
            loop_start = i;
        }
        n[8] = n[8] >> 1 | n[7] << 31;
        n[7] = n[7] >> 1 | n[6] << 31;
        n[6] = n[6] >> 1 | n[5] << 31;
        n[5] = n[5] >> 1 | n[4] << 31;
        n[4] = n[4] >> 1 | n[3] << 31;
        n[3] = n[3] >> 1 | n[2] << 31;
        n[2] = n[2] >> 1 | n[1] << 31;
        n[1] = n[1] >> 1 | n[0] << 31;
        n[0] = n[0] >> 1;
    }
    return loop_start;
}

__device__ void point_mul_xy(u32* x1, u32* y1, const u32* k, const secp256k1_t* tmps) {
    u32 naf[SECP256K1_NAF_SIZE] = { 0 };
    int loop_start = convert_to_window_naf(naf, k);
    const u32 multiplier = (naf[loop_start >> 3] >> ((loop_start & 7) << 2)) & 0x0f; // or use u8 ?
    const u32 odd = multiplier & 1;
    const u32 x_pos = ((multiplier - 1 + odd) >> 1) * 24;
    const u32 y_pos = odd ? (x_pos + 8) : (x_pos + 16);
    x1[0] = tmps->xy[x_pos + 0];
    x1[1] = tmps->xy[x_pos + 1];
    x1[2] = tmps->xy[x_pos + 2];
    x1[3] = tmps->xy[x_pos + 3];
    x1[4] = tmps->xy[x_pos + 4];
    x1[5] = tmps->xy[x_pos + 5];
    x1[6] = tmps->xy[x_pos + 6];
    x1[7] = tmps->xy[x_pos + 7];
    y1[0] = tmps->xy[y_pos + 0];
    y1[1] = tmps->xy[y_pos + 1];
    y1[2] = tmps->xy[y_pos + 2];
    y1[3] = tmps->xy[y_pos + 3];
    y1[4] = tmps->xy[y_pos + 4];
    y1[5] = tmps->xy[y_pos + 5];
    y1[6] = tmps->xy[y_pos + 6];
    y1[7] = tmps->xy[y_pos + 7];
    u32 z1[8] = { 0 };
    z1[0] = 1;
    for (int pos = loop_start - 1; pos >= 0; pos--) {
        point_double(x1, y1, z1);
        const u32 multiplier = (naf[pos >> 3] >> ((pos & 7) << 2)) & 0x0f;
        if (multiplier) {
            const u32 odd = multiplier & 1;
            const u32 x_pos = ((multiplier - 1 + odd) >> 1) * 24;
            const u32 y_pos = odd ? (x_pos + 8) : (x_pos + 16);
            u32 x2[8];
            x2[0] = tmps->xy[x_pos + 0];
            x2[1] = tmps->xy[x_pos + 1];
            x2[2] = tmps->xy[x_pos + 2];
            x2[3] = tmps->xy[x_pos + 3];
            x2[4] = tmps->xy[x_pos + 4];
            x2[5] = tmps->xy[x_pos + 5];
            x2[6] = tmps->xy[x_pos + 6];
            x2[7] = tmps->xy[x_pos + 7];
            u32 y2[8];
            y2[0] = tmps->xy[y_pos + 0];
            y2[1] = tmps->xy[y_pos + 1];
            y2[2] = tmps->xy[y_pos + 2];
            y2[3] = tmps->xy[y_pos + 3];
            y2[4] = tmps->xy[y_pos + 4];
            y2[5] = tmps->xy[y_pos + 5];
            y2[6] = tmps->xy[y_pos + 6];
            y2[7] = tmps->xy[y_pos + 7];
            point_add(x1, y1, z1, x2, y2);
        }
    }
    inv_mod(z1);
    u32 z2[8];
    mul_mod(z2, z1, z1); // z1^2
    mul_mod(x1, x1, z2); // x1_affine
    mul_mod(z1, z2, z1); // z1^3
    mul_mod(y1, y1, z1); // y1_affine
}

__device__ void set_precomputed_basepoint_g(secp256k1_t* r) {
    // x1
    r->xy[0] = SECP256K1_G_PRE_COMPUTED_00;
    r->xy[1] = SECP256K1_G_PRE_COMPUTED_01;
    r->xy[2] = SECP256K1_G_PRE_COMPUTED_02;
    r->xy[3] = SECP256K1_G_PRE_COMPUTED_03;
    r->xy[4] = SECP256K1_G_PRE_COMPUTED_04;
    r->xy[5] = SECP256K1_G_PRE_COMPUTED_05;
    r->xy[6] = SECP256K1_G_PRE_COMPUTED_06;
    r->xy[7] = SECP256K1_G_PRE_COMPUTED_07;

    // y1
    r->xy[8] = SECP256K1_G_PRE_COMPUTED_08;
    r->xy[9] = SECP256K1_G_PRE_COMPUTED_09;
    r->xy[10] = SECP256K1_G_PRE_COMPUTED_10;
    r->xy[11] = SECP256K1_G_PRE_COMPUTED_11;
    r->xy[12] = SECP256K1_G_PRE_COMPUTED_12;
    r->xy[13] = SECP256K1_G_PRE_COMPUTED_13;
    r->xy[14] = SECP256K1_G_PRE_COMPUTED_14;
    r->xy[15] = SECP256K1_G_PRE_COMPUTED_15;

    // -y1
    r->xy[16] = SECP256K1_G_PRE_COMPUTED_16;
    r->xy[17] = SECP256K1_G_PRE_COMPUTED_17;
    r->xy[18] = SECP256K1_G_PRE_COMPUTED_18;
    r->xy[19] = SECP256K1_G_PRE_COMPUTED_19;
    r->xy[20] = SECP256K1_G_PRE_COMPUTED_20;
    r->xy[21] = SECP256K1_G_PRE_COMPUTED_21;
    r->xy[22] = SECP256K1_G_PRE_COMPUTED_22;
    r->xy[23] = SECP256K1_G_PRE_COMPUTED_23;

    // x3
    r->xy[24] = SECP256K1_G_PRE_COMPUTED_24;
    r->xy[25] = SECP256K1_G_PRE_COMPUTED_25;
    r->xy[26] = SECP256K1_G_PRE_COMPUTED_26;
    r->xy[27] = SECP256K1_G_PRE_COMPUTED_27;
    r->xy[28] = SECP256K1_G_PRE_COMPUTED_28;
    r->xy[29] = SECP256K1_G_PRE_COMPUTED_29;
    r->xy[30] = SECP256K1_G_PRE_COMPUTED_30;
    r->xy[31] = SECP256K1_G_PRE_COMPUTED_31;

    // y3
    r->xy[32] = SECP256K1_G_PRE_COMPUTED_32;
    r->xy[33] = SECP256K1_G_PRE_COMPUTED_33;
    r->xy[34] = SECP256K1_G_PRE_COMPUTED_34;
    r->xy[35] = SECP256K1_G_PRE_COMPUTED_35;
    r->xy[36] = SECP256K1_G_PRE_COMPUTED_36;
    r->xy[37] = SECP256K1_G_PRE_COMPUTED_37;
    r->xy[38] = SECP256K1_G_PRE_COMPUTED_38;
    r->xy[39] = SECP256K1_G_PRE_COMPUTED_39;

    // -y3
    r->xy[40] = SECP256K1_G_PRE_COMPUTED_40;
    r->xy[41] = SECP256K1_G_PRE_COMPUTED_41;
    r->xy[42] = SECP256K1_G_PRE_COMPUTED_42;
    r->xy[43] = SECP256K1_G_PRE_COMPUTED_43;
    r->xy[44] = SECP256K1_G_PRE_COMPUTED_44;
    r->xy[45] = SECP256K1_G_PRE_COMPUTED_45;
    r->xy[46] = SECP256K1_G_PRE_COMPUTED_46;
    r->xy[47] = SECP256K1_G_PRE_COMPUTED_47;

    // x5
    r->xy[48] = SECP256K1_G_PRE_COMPUTED_48;
    r->xy[49] = SECP256K1_G_PRE_COMPUTED_49;
    r->xy[50] = SECP256K1_G_PRE_COMPUTED_50;
    r->xy[51] = SECP256K1_G_PRE_COMPUTED_51;
    r->xy[52] = SECP256K1_G_PRE_COMPUTED_52;
    r->xy[53] = SECP256K1_G_PRE_COMPUTED_53;
    r->xy[54] = SECP256K1_G_PRE_COMPUTED_54;
    r->xy[55] = SECP256K1_G_PRE_COMPUTED_55;

    // y5
    r->xy[56] = SECP256K1_G_PRE_COMPUTED_56;
    r->xy[57] = SECP256K1_G_PRE_COMPUTED_57;
    r->xy[58] = SECP256K1_G_PRE_COMPUTED_58;
    r->xy[59] = SECP256K1_G_PRE_COMPUTED_59;
    r->xy[60] = SECP256K1_G_PRE_COMPUTED_60;
    r->xy[61] = SECP256K1_G_PRE_COMPUTED_61;
    r->xy[62] = SECP256K1_G_PRE_COMPUTED_62;
    r->xy[63] = SECP256K1_G_PRE_COMPUTED_63;

    // -y5
    r->xy[64] = SECP256K1_G_PRE_COMPUTED_64;
    r->xy[65] = SECP256K1_G_PRE_COMPUTED_65;
    r->xy[66] = SECP256K1_G_PRE_COMPUTED_66;
    r->xy[67] = SECP256K1_G_PRE_COMPUTED_67;
    r->xy[68] = SECP256K1_G_PRE_COMPUTED_68;
    r->xy[69] = SECP256K1_G_PRE_COMPUTED_69;
    r->xy[70] = SECP256K1_G_PRE_COMPUTED_70;
    r->xy[71] = SECP256K1_G_PRE_COMPUTED_71;

    // x7
    r->xy[72] = SECP256K1_G_PRE_COMPUTED_72;
    r->xy[73] = SECP256K1_G_PRE_COMPUTED_73;
    r->xy[74] = SECP256K1_G_PRE_COMPUTED_74;
    r->xy[75] = SECP256K1_G_PRE_COMPUTED_75;
    r->xy[76] = SECP256K1_G_PRE_COMPUTED_76;
    r->xy[77] = SECP256K1_G_PRE_COMPUTED_77;
    r->xy[78] = SECP256K1_G_PRE_COMPUTED_78;
    r->xy[79] = SECP256K1_G_PRE_COMPUTED_79;

    // y7
    r->xy[80] = SECP256K1_G_PRE_COMPUTED_80;
    r->xy[81] = SECP256K1_G_PRE_COMPUTED_81;
    r->xy[82] = SECP256K1_G_PRE_COMPUTED_82;
    r->xy[83] = SECP256K1_G_PRE_COMPUTED_83;
    r->xy[84] = SECP256K1_G_PRE_COMPUTED_84;
    r->xy[85] = SECP256K1_G_PRE_COMPUTED_85;
    r->xy[86] = SECP256K1_G_PRE_COMPUTED_86;
    r->xy[87] = SECP256K1_G_PRE_COMPUTED_87;

    // -y7
    r->xy[88] = SECP256K1_G_PRE_COMPUTED_88;
    r->xy[89] = SECP256K1_G_PRE_COMPUTED_89;
    r->xy[90] = SECP256K1_G_PRE_COMPUTED_90;
    r->xy[91] = SECP256K1_G_PRE_COMPUTED_91;
    r->xy[92] = SECP256K1_G_PRE_COMPUTED_92;
    r->xy[93] = SECP256K1_G_PRE_COMPUTED_93;
    r->xy[94] = SECP256K1_G_PRE_COMPUTED_94;
    r->xy[95] = SECP256K1_G_PRE_COMPUTED_95;
}

__global__ void precomputeG_kernel(u32* Gx, u32* Gy) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > PRE_G_SIZE) return;
    secp256k1_t preG;
    set_precomputed_basepoint_g(&preG);
    u32 x[8];
    u32 y[8];
    if (idx < PRE_G_SIZE) {
        const u32 k[8] = { 1 + idx, 0, 0, 0, 0, 0, 0, 0 };
        point_mul_xy(x, y, k, &preG);
    } 
    if (idx == PRE_G_SIZE) {
        u32 s[8] = { SIZE, 0, 0, 0, 0, 0, 0, 0 };
        u32 g[8] = { GROUPS, 0, 0, 0, 0, 0, 0, 0 };
        u32 t[8] = { THREADS, 0, 0, 0, 0, 0, 0, 0 };
        mul_mod(s, s, g);
        mul_mod(s, s, t);
        point_mul_xy(x, y, s, &preG);
    }
    for (int i = 0; i < 8; i++) {
        (&Gx[8 * idx])[i] = x[i];
        (&Gy[8 * idx])[i] = y[i];
    }
    
}

__global__ void precompute_kernel(const uint32_t* sk, uint32_t* Cx, uint32_t* Cy) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    const u32 center[8] = { PRE_G_SIZE, 0, 0, 0, 0, 0, 0, 0 };
    const u32 offset[8] = { SIZE, 0, 0, 0, 0, 0, 0, 0 };
    const u32 i[8] = { idx, 0, 0, 0, 0, 0, 0, 0 };
    u32 x[8];
    u32 y[8];
    u32 k[8];
    mul_mod(k, offset, i);
    add(k, k, center);
    add(k, k, sk);
    secp256k1_t preG;
    set_precomputed_basepoint_g(&preG);
    point_mul_xy(x, y, k, &preG);
    for (int i = 0; i < 8; i++) {
        (&Cx[8 * idx])[i] = x[i];
        (&Cy[8 * idx])[i] = y[i];
    }
}

int precomputeG(u32* Gx, u32* Gy, u32 gpu) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(gpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    u32* dev_gx = 0;
    u32* dev_gy = 0;
    cudaStatus = cudaMalloc((void**)&(dev_gx), 8 * (PRE_G_SIZE + 1) * sizeof(u32));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaStatus = cudaMalloc((void**)&dev_gy, 8 * (PRE_G_SIZE + 1) * sizeof(u32));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    precomputeG_kernel <<< ((PRE_G_SIZE + 1) / 128) + 1, 128 >>> (dev_gx, dev_gy);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    cudaMemcpy(Gx, dev_gx, 8 * (PRE_G_SIZE + 1) * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gy, dev_gy, 8 * (PRE_G_SIZE + 1) * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaFree(dev_gx);
    cudaFree(dev_gy);
    return 0;
}

int calculateCenterPoints(u32* cx, u32* cy, const u32* sk, u32 gpu) {
#define S (GROUPS * THREADS)
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(gpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    u32* dev_sk = 0;
    u32* dev_cx = 0;
    u32* dev_cy = 0;
    cudaStatus = cudaMalloc((void**)&dev_sk, 8 * sizeof(u32));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaStatus = cudaMalloc((void**)&dev_cx, 8 * S * sizeof(u32));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaStatus = cudaMalloc((void**)&dev_cy, 8 * S * sizeof(u32));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaStatus = cudaMemcpy(dev_sk, sk, 8 * sizeof(u32), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    precompute_kernel<<<GROUPS, THREADS>>>(dev_sk, dev_cx, dev_cy);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    cudaMemcpy(cx, dev_cx, 8 * S * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(cy, dev_cy, 8 * S * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaFree(dev_cx);
    cudaFree(dev_cy);
    return 0;
}



