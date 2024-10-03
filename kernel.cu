
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "secp256k1.h"
#include "hash.h"
#include <memory.h>
#include <string.h>
#include "def.h"

#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

#define NBBLOCK 5
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b) : "memory");
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b) : "memory");
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory");
#define MADDS(r,a,b,c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory");

#define MM64 0xD838091DD2253531ULL

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define __sright128(a,b,n) ((a)>>(n))|((b)<<(64-(n)))
#define __sleft128(a,b,n) ((b)<<(n))|((a)>>(64-(n)))

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

#define Sub2(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]);}

#define Sub1(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], a[4]);}

#define Neg(r) {\
USUBO(r[0],0ULL,r[0]); \
USUBC(r[1],0ULL,r[1]); \
USUBC(r[2],0ULL,r[2]); \
USUBC(r[3],0ULL,r[3]); \
USUB(r[4],0ULL,r[4]); }

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

#define _LoadI64(r, a) {\
  (r)[0] = a; \
  (r)[1] = a>>63; \
  (r)[2] = (r)[1]; \
  (r)[3] = (r)[1]; \
  (r)[4] = (r)[1];}

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}


#define ShiftR62(r) {\
    (r)[0] = ((r)[1] << 2) | ((r)[0] >> 62); \
    (r)[1] = ((r)[2] << 2) | ((r)[1] >> 62); \
    (r)[2] = ((r)[3] << 2) | ((r)[2] >> 62); \
    (r)[3] = ((r)[4] << 2) | ((r)[3] >> 62); \
    (r)[4] = (int64_t)((r)[4]) >> 62; }

#define ShiftR62C(dest, r, carry) { \
    (dest)[0] = ((r)[1] << 2) | ((r)[0] >> 62); \
    (dest)[1] = ((r)[2] << 2) | ((r)[1] >> 62); \
    (dest)[2] = ((r)[3] << 2) | ((r)[2] >> 62); \
    (dest)[3] = ((r)[4] << 2) | ((r)[3] >> 62); \
    (dest)[4] = ((carry)<< 2) | ((r)[4] >> 62); }

__device__  void IMult(uint64_t* r, const uint64_t* a, int64_t b, uint64_t *t) {
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    }
    else {
        Load(t, a);
    }
    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADD(r[4], t[3], b, r[4]);
}

__device__ __forceinline__  uint64_t IMultC(uint64_t* r, const uint64_t* a, int64_t b, uint64_t* t) {
    uint64_t carry;
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }
    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADDC(r[4], t[3], b, r[4]);
    MADDS(carry, t[4], b, 0ULL);
    return carry;
}

#define MulP(r, a, ah, al) {\
UMULLO((al), (a), 0x1000003D1ULL); \
UMULHI((ah), (a), 0x1000003D1ULL); \
USUBO((r)[0], 0ULL, (al)); \
USUBC((r)[1], 0ULL, (ah)); \
USUBC((r)[2], 0ULL, 0ULL); \
USUBC((r)[3], 0ULL, 0ULL); \
USUB((r)[4], (a), 0ULL); }

#define ModNeg256(r, a, t) { \
USUBO((t)[0], 0ULL, (a)[0]); \
USUBC((t)[1], 0ULL, (a)[1]); \
USUBC((t)[2], 0ULL, (a)[2]); \
USUBC((t)[3], 0ULL, (a)[3]); \
UADDO((r)[0], (t)[0], 0xFFFFFFFEFFFFFC2FULL); \
UADDC((r)[1], (t)[1], 0xFFFFFFFFFFFFFFFFULL); \
UADDC((r)[2], (t)[2], 0xFFFFFFFFFFFFFFFFULL); \
UADD((r)[3], (t)[3], 0xFFFFFFFFFFFFFFFFULL); }


__device__ __forceinline__ void ModSub256(uint64_t* r, const uint64_t* a, const uint64_t* b, uint64_t *T) {
    uint64_t t;
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);
}

#define ctz(x) __clzll(__brevll(x))
#define SWAP(tmp,x,y) tmp = x; x = y; y = tmp;
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ __forceinline__ void _DivStep62(const uint64_t *u, const uint64_t *v,
    int32_t* pos,
    int64_t* uu, int64_t* uv,
    int64_t* vu, int64_t* vv) {
    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;
    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];
    uint64_t uh, vh;
    int64_t w;
    bitCount = 62;
    while (*pos > 0 && (u[*pos] | v[*pos]) == 0) (*pos)--;
    if (*pos == 0) {
        uh = u[0];
        vh = v[0];
    } else {
        uint32_t s = __clzll(u[*pos] | v[*pos]);
        if (s == 0) {
            uh = u[*pos];
            vh = v[*pos];
        } else {
            uh = __sleft128(u[*pos - 1], u[*pos], s);
            vh = __sleft128(v[*pos - 1], v[*pos], s);
        }
    }
    while (true) {
        zeros = ctz(v0 | (1ULL << bitCount));
        v0 >>= zeros;
        vh >>= zeros;
        *uu <<= zeros;
        *uv <<= zeros;
        bitCount -= zeros;
        if (bitCount == 0) break;
        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(w, u0, v0);
            SWAP(w, *uu, *vu);
            SWAP(w, *uv, *vv);
        }
        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;
    }
}

__device__  __forceinline__ void MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, uint64_t* carry, uint64_t *t1, uint64_t *t2, uint64_t* t3) {
    uint64_t c1, c2;
    c1 = IMultC(t1, u, _11, t3);
    c2 = IMultC(t2, v, _12, t3);
    UADDO(dest[0], t1[0], t2[0]);
    UADDC(dest[1], t1[1], t2[1]);
    UADDC(dest[2], t1[2], t2[2]);
    UADDC(dest[3], t1[3], t2[3]);
    UADDC(dest[4], t1[4], t2[4]);
    UADD(*carry, c1, c2);

}

__device__ __forceinline__  void MatrixVecMul(uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, int64_t _21, int64_t _22, uint64_t* t1, uint64_t* t2, uint64_t* t3, uint64_t* t4, uint64_t* t5) {
    IMult(t1, u, _11, t5);
    IMult(t2, v, _12, t5);
    IMult(t3, u, _21, t5);
    IMult(t4, v, _22, t5);
    UADDO(u[0], t1[0], t2[0]);
    UADDC(u[1], t1[1], t2[1]);
    UADDC(u[2], t1[2], t2[2]);
    UADDC(u[3], t1[3], t2[3]);
    UADD(u[4], t1[4], t2[4]);
    UADDO(v[0], t3[0], t4[0]);
    UADDC(v[1], t3[1], t4[1]);
    UADDC(v[2], t3[2], t4[2]);
    UADDC(v[3], t3[3], t4[3]);
    UADD(v[4], t3[4], t4[4]);
}

#define AddCh(r, a, carry) {\
UADDO1(r[0], a[0]); \
UADDC1(r[1], a[1]); \
UADDC1(r[2], a[2]); \
UADDC1(r[3], a[3]); \
UADDC1(r[4], a[4]); \
UADD(carry, carry, 0ULL); }


__device__ void _ModInv(uint64_t* v) {
    int64_t  uu, uv, vu, vv;
    uint64_t mr0, ms0, ah, al;
    int32_t  pos = NBBLOCK - 1;
    uint64_t u[NBBLOCK];
    uint64_t r[NBBLOCK];
    uint64_t s[NBBLOCK];
    uint64_t tr[NBBLOCK];
    uint64_t ts[NBBLOCK];
    uint64_t r0[NBBLOCK];
    uint64_t s0[NBBLOCK];
    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t t3[NBBLOCK];
    uint64_t t4[NBBLOCK];
    uint64_t t5[NBBLOCK];
    uint64_t carryR;
    uint64_t carryS;
    u[0] = 0xFFFFFFFEFFFFFC2F;
    u[1] = 0xFFFFFFFFFFFFFFFF;
    u[2] = 0xFFFFFFFFFFFFFFFF;
    u[3] = 0xFFFFFFFFFFFFFFFF;
    u[4] = 0;
    //Load(v, R);
    r[0] = 0; s[0] = 1;
    r[1] = 0; s[1] = 0;
    r[2] = 0; s[2] = 0;
    r[3] = 0; s[3] = 0;
    r[4] = 0; s[4] = 0;
    while (true) {
        _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);
        MatrixVecMul(u, v, uu, uv, vu, vv, t1, t2, t3, t4, t5);
        if (_IsNegative(u)) {
            Neg(u);
            uu = -uu;
            uv = -uv;
        }
        if (_IsNegative(v)) {
            Neg(v);
            vu = -vu;
            vv = -vv;
        }
        ShiftR62(u);
        ShiftR62(v);
        MatrixVecMulHalf(tr, r, s, uu, uv, &carryR, t1, t2, t3);
        mr0 = (tr[0] * MM64) & MSK62;
        MulP(r0, mr0, ah, al);
        AddCh(tr, r0, carryR);
        if (_IsZero(v)) {
            ShiftR62C(r, tr, carryR);
            break;
        } else {
            MatrixVecMulHalf(ts, r, s, vu, vv, &carryS, t1, t2, t3);
            ms0 = (ts[0] * MM64) & MSK62;
            MulP(s0, ms0, ah, al);
            AddCh(ts, s0, carryS);
        }
        ShiftR62C(r, tr, carryR);
        ShiftR62C(s, ts, carryS);
    }
   /* if (!_IsOne(u)) {
        v[0] = 0ULL;
        v[1] = 0ULL;
        v[2] = 0ULL;
        v[3] = 0ULL;
        v[4] = 0ULL;
        return;
    }*/
    while (_IsNegative(r))
        AddP(r);
    while (!_IsNegative(r))
        SubP(r);
    AddP(r);
    Load(v, r);
}

__device__ __forceinline__  void _ModMult(uint64_t* r, const uint64_t* a, const uint64_t* b, uint64_t *r512, uint64_t *t) {
    uint64_t ah, al;
    r512[5] = 0;
    r512[6] = 0;
    r512[7] = 0;
    UMult(r512, a, b[0]);
    UMult(t, a, b[1]);
    UADDO1(r512[1], t[0]);
    UADDC1(r512[2], t[1]);
    UADDC1(r512[3], t[2]);
    UADDC1(r512[4], t[3]);
    UADD1(r512[5], t[4]);
    UMult(t, a, b[2]);
    UADDO1(r512[2], t[0]);
    UADDC1(r512[3], t[1]);
    UADDC1(r512[4], t[2]);
    UADDC1(r512[5], t[3]);
    UADD1(r512[6], t[4]);
    UMult(t, a, b[3]);
    UADDO1(r512[3], t[0]);
    UADDC1(r512[4], t[1]);
    UADDC1(r512[5], t[2]);
    UADDC1(r512[6], t[3]);
    UADD1(r512[7], t[4]);
    UMult(t, (r512 + 4), 0x1000003D1ULL);
    UADDO1(r512[0], t[0]);
    UADDC1(r512[1], t[1]);
    UADDC1(r512[2], t[2]);
    UADDC1(r512[3], t[3]);
    UADD1(t[4], 0ULL);
    UMULLO(al, t[4], 0x1000003D1ULL);
    UMULHI(ah, t[4], 0x1000003D1ULL);
    UADDO(r[0], r512[0], al);
    UADDC(r[1], r512[1], ah);
    UADDC(r[2], r512[2], 0ULL);
    UADD(r[3], r512[3], 0ULL);
}

__constant__ uint8_t addr_hash[20];

__device__ __forceinline__  bool check_key(const uint64_t* x, const uint64_t* y, uint32_t* pk, uint8_t* hash) {
    /*uint8_t hash[20] = { 0 };*/
    //GetAddrHashUncomp(x, y, hash);
    GetAddrHashComp(x, y[0] % 2, hash, pk);
#pragma unroll 20
    for (uint32_t i = 0; i < 20; i++) {
        //if (addr_hash[i] != hash[i] && addr_hash[i] != hash[20 + i]) return false;
        if (addr_hash[i] != hash[i]) return false;
    }
    return true;
}

#define setRes(res, idx, i) {\
res[0] = 1; \
res[1] = idx; \
res[2] = i; }



__constant__ uint64_t preGx[4 * (PRE_G_SIZE + 1)];
__constant__ uint64_t preGy[4 * (PRE_G_SIZE + 1)];


__global__ void brute_kernel(uint32_t* res, uint64_t* Cx, uint64_t* Cy) {
    uint64_t dx[(PRE_G_SIZE + 1) * 4], dy[4], subp[(PRE_G_SIZE + 1) * 4], pyn[5], px[4], py[4], s[4], p[4];
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t i;
    uint32_t publicKeyBytes[16];
    uint8_t hash[20] = { 0 };
        uint64_t r512[8];
    uint64_t t[NBBLOCK];

#define sx (&Cx[4 * idx])
#define sy (&Cy[4 * idx])

    Load256(px, sx);
    Load256(py, sy);
    if (check_key(px, py, publicKeyBytes, hash)) {
        setRes(res, idx, PRE_G_SIZE);
    }

    for (i = 0; i < (PRE_G_SIZE + 1); i++) {
        ModSub256(&dx[4 * i], &preGx[4 * i], sx, t);
    }

    Load256(&subp[0], &dx[0]);
    for (i = 1; i < (PRE_G_SIZE + 1); i++) {
        _ModMult(&subp[4 * i], &subp[4 * (i - 1)], &dx[4 * i], r512, t);
    }
    Load256(pyn, &subp[4 * PRE_G_SIZE]);
    pyn[4] = 0;
    _ModInv(pyn);
    for (i = PRE_G_SIZE; i > 0; i--) {
        _ModMult(s, &subp[4 * (i - 1)], pyn, r512, t);
        _ModMult(pyn, pyn, &dx[4 * i], r512, t);
        Load256(&dx[4 * i], s);
    }
    Load256(dx, pyn);

    ModNeg256(pyn, py, t);


    for (i = 0; i < PRE_G_SIZE; i++) {
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, &preGy[i * 4], py, t);      
        _ModMult(s, dy, &dx[i * 4], r512, t);
        _ModMult(p, s, s, r512, t);
        ModSub256(px, p, px, t);
        ModSub256(px, px, &preGx[i * 4], t);
        ModSub256(py, &preGx[i * 4], px, t);
        _ModMult(py, py, s, r512, t);
        ModSub256(py, py, &preGy[i * 4], t);
        if (check_key(px, py, publicKeyBytes, hash)) {
            setRes(res, idx, PRE_G_SIZE + i + 1);
        }

        Load256(px, sx);
        ModSub256(dy, pyn, &preGy[i * 4], t);
        _ModMult(s, dy, &dx[i * 4], r512, t);
        _ModMult(p, s, s, r512, t);
        ModSub256(px, p, px, t);
        ModSub256(px, px, &preGx[i * 4], t);
        ModSub256(py, px, &preGx[i * 4], t);
        _ModMult(py, py, s, r512, t);
        ModSub256(py, &preGy[i * 4], py, t);
        if (check_key(px, py, publicKeyBytes, hash)) {
            setRes(res, idx, PRE_G_SIZE - i - 1);
        }
    }

    Load256(px, sx);
    Load256(py, sy);
    ModSub256(dy, &preGy[4 * PRE_G_SIZE], py, t);
    _ModMult(s, dy, &dx[4 * PRE_G_SIZE], r512, t);
    _ModMult(p, s, s, r512, t);
    ModSub256(px, p, px, t);
    ModSub256(px, px, &preGx[4 * PRE_G_SIZE], t);
    ModSub256(py, &preGx[4 * PRE_G_SIZE], px, t);
    _ModMult(py, py, s, r512, t);
    ModSub256(py, py, &preGy[4 * PRE_G_SIZE], t);

    Load256(sx, px);
    Load256(sy, py);
#undef sx
#undef sy
}


// ------------------------------------------------------------------------------------------------------------------


void sprint256(char* r, const uint32_t* a) {
    for (int i = 0; i < 8; i++) {
        sprintf(r + 8 * i, "%08x", a[7 - i]);
    }
    r[64] = 0;
}

void parse_addr_hash(uint8_t *r, const char* h) {
    for (int i = 0; i < 20; i++) {
        char tmp[3];
        tmp[2] = 0;
        for (int j = 0; j < 2; j++) {
            tmp[j] = h[i * 2 + j];
        }
        r[i] = (uint8_t)strtoul(tmp, NULL, 16);
    }
}

void read256(uint32_t* k, const char* str) {
    size_t ilen = strlen(str);
    size_t i = 0;
    while (ilen > 0) {
        char tmp[9] = { 0 };
        size_t len = ilen > 8 ? 8 : ilen;
        ilen -= len;
        memcpy(tmp, str + ilen, len);
        k[i++] = strtoul(tmp, NULL, 16);
    }
}

int main(int argc, char* argv[]) {
    char tmp[65];
    char filename[200];
    char bk_filename[200];
    uint32_t sk[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    uint8_t hash[20];
    uint32_t gpu_n = 0;
    bool restore = false;
    u32* res = (u32*)malloc(3 * sizeof(u32));
    u32* Gx = (u32*)malloc(8 * sizeof(u32) * (PRE_G_SIZE + 1));
    u32* Gy = (u32*)malloc(8 * sizeof(u32) * (PRE_G_SIZE + 1));
    u32* cx = (u32*)malloc(8 * sizeof(u32) * THREADS * GROUPS);
    u32* cy = (u32*)malloc(8 * sizeof(u32) * THREADS * GROUPS);
    FILE* out_f = 0;
    if (argc > 1) {
        read256(sk, argv[1]);
    }
    if (argc > 2) {
        parse_addr_hash(hash, argv[2]);
    }
    if (argc > 3) {
        gpu_n = strtoul(argv[3], NULL, 10);
    }
    sprint256(filename, sk);
    sprintf(filename + strlen(filename), ".txt");

    sprintf(bk_filename, ".");
    sprint256(bk_filename + 1, sk);
    sprintf(bk_filename + strlen(bk_filename), "-bk");
    if (access(bk_filename, F_OK) == 0) {
        FILE* bkf = fopen(bk_filename, "r");
        char str_sk[65];
        fread(str_sk, 1, 64, bkf);
        str_sk[64] = 0;
        read256(sk, str_sk);
        fclose(bkf);
        restore = true;
    }
    if (!restore) {
        out_f = fopen(filename, "w");
        fprintf(out_f, "chunk size: %llu\n", (uint64_t)THREADS * (uint64_t)GROUPS * (uint64_t)SIZE);
        fprintf(out_f, "addr hash: ");
        for (int i = 0; i < 20; i++) fprintf(out_f, "%02x", hash[i]);
        fprintf(out_f, "\n");
        sprint256(tmp, sk);
        fprintf(out_f, "start key: %s\n", tmp);
        fclose(out_f);
    }

    precomputeG(Gx, Gy, gpu_n);
    calculateCenterPoints(cx, cy, sk, gpu_n);

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(gpu_n);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }

    cudaStatus = cudaMemcpyToSymbol(preGx, Gx, 4 * (PRE_G_SIZE + 1) * sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
    cudaStatus = cudaMemcpyToSymbol(preGy, Gy, 4 * (PRE_G_SIZE + 1) * sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
 
    cudaStatus = cudaMemcpyToSymbol(addr_hash, hash, 20);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }

    uint32_t* dev_res = 0;
    uint64_t* dev_Cx = 0;
    uint64_t* dev_Cy = 0;

    cudaStatus = cudaMalloc((void**)&dev_res, 3 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
    cudaStatus = cudaMalloc((void**)&dev_Cx, 4 * THREADS * GROUPS * sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
    cudaStatus = cudaMalloc((void**)&dev_Cy, 4 * THREADS * GROUPS * sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);;
    }

    cudaStatus = cudaMemcpy(dev_Cx, cx, 4 * THREADS * GROUPS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }

    cudaStatus = cudaMemcpy(dev_Cy, cy, 4 * THREADS * GROUPS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }

    uint32_t chunk[8] = { SIZE, 0, 0, 0, 0, 0, 0 };
    uint32_t checked_keys[8] = { THREADS * GROUPS, 0, 0, 0, 0 };
    mul_mod(chunk, chunk, checked_keys);
    for (int i = 0; i < 8; i++) checked_keys[i] = sk[i];
    clock_t glb_start = clock();
    clock_t lcl_start;

    free(Gx);
    free(Gy);
    free(cx);
    free(cy);
    
    while (1) {
        lcl_start = clock();
        brute_kernel << <GROUPS, THREADS >> > (dev_res, dev_Cx, dev_Cy);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        cudaMemcpy(res, dev_res, 3 * sizeof(u32), cudaMemcpyDeviceToHost);
        if (res[0] == 1) {
            uint32_t t[8] = { SIZE, 0, 0, 0, 0, 0, 0 };
            uint32_t offset[8] = { res[1], 0, 0, 0, 0, 0, 0, 0 };
            mul_mod(offset, offset, t);
            t[0] = res[2];
            add(offset, offset, t);
            add(offset, offset, checked_keys);
            sprint256(tmp, offset);
            out_f = fopen(filename, "a");
            fprintf(out_f, "FOUND PRIVATE KEY: %s\n", tmp);
            fclose(out_f);
            exit(0);
        }

        add(checked_keys, checked_keys, chunk);
        clock_t end = clock();
        double lcl_time = (double)(end - lcl_start) / CLOCKS_PER_SEC;
        double speed = (double)((uint64_t)THREADS * (uint64_t)GROUPS * (uint64_t)SIZE) / lcl_time;
        double glbl_time = (double)(end - glb_start) / CLOCKS_PER_SEC;
        sprint256(tmp, checked_keys);
        out_f = fopen(filename, "a");
        fprintf(out_f, "%s time: %lf speed: %lf/s\n", tmp, glbl_time, speed);
        fclose(out_f);

        FILE* bkf = fopen(bk_filename, "w");
        fprintf(bkf, "%s", tmp);
        fclose(bkf);
    }
    cudaFree(dev_res);
    cudaFree(dev_Cx);
    cudaFree(dev_Cy);
    return 0;
}