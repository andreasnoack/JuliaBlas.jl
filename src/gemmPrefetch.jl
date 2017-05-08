module LC

const MC = 384
const KC = 384
const NC = 4096

const MR = 4
const NR = 4

const _A = Array{Float64}(MC*KC)
const _B = Array{Float64}(KC*NC)
const _C = Array{Float64}(MR*NR)

const _AB = Array{Float64}(MR*NR)

function pack_MRxk(k::Integer, A::Array{Float64}, Aoffset::Integer, incRowA::Integer,
    incColA::Integer, buffer::Array{Float64}, boffset::Integer)
@inbounds begin
    for j = 1:k
        @simd for i = 1:MR
            buffer[boffset + i] = A[Aoffset + (i - 1)*incRowA + 1]
        end
        boffset += MR
        Aoffset += incColA
    end
end
end

function pack_A(mc::Integer, kc::Integer, A::Array{Float64}, Aoffset::Integer,
    incRowA::Integer, incColA::Integer, buffer::Array{Float64})
@inbounds begin
    mp  = div(mc, MR)
    _mr = mc % MR

    boffset = 0
    for i = 1:mp
        pack_MRxk(kc, A, Aoffset, incRowA, incColA, buffer, boffset)
        boffset += kc*MR
        Aoffset += MR*incRowA
    end
    if _mr > 0
        for j = 1:kc
            for i = 1:_mr
                buffer[boffset + i] = A[Aoffset + (i - 1)*incRowA + 1]
            end
            for i = _mr:MR
                buffer[boffset + i] = 0.0
            end
            boffset += MR
            Aoffset += incColA
        end
    end
end
end

function pack_kxNR(k::Integer, B::Array{Float64}, Boffset::Integer, incRowB::Integer,
    incColB::Integer, buffer::Array{Float64}, boffset::Integer)
@inbounds begin
    for i = 1:k
        for j = 1:NR
            buffer[boffset + j] = B[Boffset + (j - 1)*incColB + 1]
        end
        boffset += NR
        Boffset += incRowB
    end
end
end
function pack_B(kc::Integer, nc::Integer, B::Array{Float64}, Boffset::Integer, incRowB::Integer,
    incColB::Integer, buffer::Array{Float64})
@inbounds begin
    np  = div(nc, NR)
    _nr = nc % NR

    boffset = 0
    for j = 1:np
        pack_kxNR(kc, B, Boffset, incRowB, incColB, buffer, boffset)
        boffset += kc*NR
        Boffset += NR*incColB
    end
    if _nr > 0
        for i = 1:kc
            for j = 1:_nr
                buffer[boffset + j] = B[Boffset + (j - 1)*incColB + 1]
            end
            for j = _nr + 1:NR
                buffer[boffset + j] = 0.0
            end
            boffset += NR
            Boffset += incRowB
        end
    end
end
end

const asms =
  "\"movq      \$0,     %rsi    \n\t"*  # kb (32 bit) stored in %rsi
    "movq      \$1,     %rdi    \n\t"*  # kl (32 bit) stored in %rdi
    "movq      \$2,     %rax    \n\t"*  # Address of A stored in %rax
    "movq      \$3,     %rbx    \n\t"*  # Address of B stored in %rbx
    "movq      \$9,     %r9     \n\t"*  # Address of nextA stored in %r9
    "movq      \$10,    %r10    \n\t"*  # Address of nextB stored in %r10
    "                            \n\t"*
    "movapd    (%rax), %xmm0   \n\t"*  # tmp0 = _mm_load_pd(A)
    "movapd  16(%rax), %xmm1   \n\t"*  # tmp1 = _mm_load_pd(A+2)
    "movapd    (%rbx), %xmm2   \n\t"*  # tmp2 = _mm_load_pd(B)
    "                            \n\t"*
    "xorpd     %xmm8,  %xmm8   \n\t"*  # ab_00_11 = _mm_setzero_pd()
    "xorpd     %xmm9,  %xmm9   \n\t"*  # ab_20_31 = _mm_setzero_pd()
    "xorpd     %xmm10, %xmm10  \n\t"*  # ab_01_10 = _mm_setzero_pd()
    "xorpd     %xmm11, %xmm11  \n\t"*  # ab_21_30 = _mm_setzero_pd()
    "xorpd     %xmm12, %xmm12  \n\t"*  # ab_02_13 = _mm_setzero_pd()
    "xorpd     %xmm13, %xmm13  \n\t"*  # ab_22_33 = _mm_setzero_pd()
    "xorpd     %xmm14, %xmm14  \n\t"*  # ab_03_12 = _mm_setzero_pd()
    "xorpd     %xmm15, %xmm15  \n\t"*  # ab_23_32 = _mm_setzero_pd()
    "                            \n\t"*
    "xorpd     %xmm3,  %xmm3   \n\t"*  # tmp3 = _mm_setzero_pd
    "xorpd     %xmm4,  %xmm4   \n\t"*  # tmp4 = _mm_setzero_pd
    "xorpd     %xmm5,  %xmm5   \n\t"*  # tmp5 = _mm_setzero_pd
    "xorpd     %xmm6,  %xmm6   \n\t"*  # tmp6 = _mm_setzero_pd
    "xorpd     %xmm7,  %xmm7   \n\t"*  # tmp7 = _mm_setzero_pd
    "testq     %rdi,   %rdi    \n\t"*  # if kl==0 writeback to AB
    "                            \n\t"*
    "                            \n\t"*
    "testq     %rsi,   %rsi    \n\t"*  # if kb==0 handle remaining kl
    # Adding ${:uid} at the end of label can fix the error
    # ```
    # error: invalid symbol redefinition
    # LLVM ERROR: Error parsing inline asm
    # ```
    # as referenced in
    # http://llvm.org/docs/LangRef.html#inline-assembler-expressions
    "je        .DCONSIDERLEFT\${:uid}  \n\t"*  # update iterations
    "                            \n\t"*
    ".DLOOP\${:uid}:                   \n\t"*  # for l = kb,..,1 do
    "                            \n\t"*
    "prefetcht0 (4*39+1)*8(%rax)\n\t"*
    "                            \n\t"*
    "                            \n\t"*  # 1. update
    "addpd     %xmm3,  %xmm12  \n\t"*  # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"*  # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"*  # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"*  # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"*  # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"*  # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm5,  %xmm14  \n\t"*  # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"*  # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"*  # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"*  # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm2,  %xmm8   \n\t"*  # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"*  # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"*  # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"*  # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"*  # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"*  # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm4,  %xmm10  \n\t"*  # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"*  # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"*  # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"*  # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"*  # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"*  # tmp1     = _mm_load_pd(A+6)
    "                            \n\t"*
    "                            \n\t"*
    "                            \n\t"*
    "                            \n\t"*  # 2. update
    "addpd     %xmm3,  %xmm12  \n\t"*  # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  48(%rbx), %xmm3   \n\t"*  # tmp3     = _mm_load_pd(B+6)
    "addpd     %xmm6,  %xmm13  \n\t"*  # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"*  # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"*  # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"*  # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm5,  %xmm14  \n\t"*  # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"*  # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"*  # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"*  # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm2,  %xmm8   \n\t"*  # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  64(%rbx), %xmm2   \n\t"*  # tmp2     = _mm_load_pd(B+8)
    "addpd     %xmm6,  %xmm9   \n\t"*  # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"*  # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"*  # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"*  # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm4,  %xmm10  \n\t"*  # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"*  # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"*  # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"*  # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  64(%rax), %xmm0   \n\t"*  # tmp0     = _mm_load_pd(A+8)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  80(%rax), %xmm1   \n\t"*  # tmp1     = _mm_load_pd(A+10)
    "                            \n\t"*
    "                            \n\t"*
    "prefetcht0 (4*41+1)*8(%rax)\n\t"*
    "                            \n\t"*
    "                            \n\t"*  # 3. update
    "addpd     %xmm3,  %xmm12  \n\t"*  # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  80(%rbx), %xmm3   \n\t"*  # tmp3     = _mm_load_pd(B+10)
    "addpd     %xmm6,  %xmm13  \n\t"*  # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"*  # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"*  # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"*  # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm5,  %xmm14  \n\t"*  # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"*  # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"*  # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"*  # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm2,  %xmm8   \n\t"*  # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  96(%rbx), %xmm2   \n\t"*  # tmp2     = _mm_load_pd(B+12)
    "addpd     %xmm6,  %xmm9   \n\t"*  # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"*  # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"*  # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"*  # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm4,  %xmm10  \n\t"*  # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"*  # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"*  # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"*  # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  96(%rax), %xmm0   \n\t"*  # tmp0     = _mm_load_pd(A+12)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd 112(%rax), %xmm1   \n\t"*  # tmp1     = _mm_load_pd(A+14)
    "                            \n\t"*
    "                            \n\t"*
    "                            \n\t"*  # 4. update
    "addpd     %xmm3,  %xmm12  \n\t"*  # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd 112(%rbx), %xmm3   \n\t"*  # tmp3     = _mm_load_pd(B+14)
    "addpd     %xmm6,  %xmm13  \n\t"*  # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"*  # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"*  # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"*  # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"*
    "addq      \$\$32*4,   %rax    \n\t"*  # A += 16;
    "                            \n\t"*
    "addpd     %xmm5,  %xmm14  \n\t"*  # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"*  # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"*  # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"*  # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"*
    "addq      \$\$128,    %r9     \n\t"*  # nextB += 16
    "                            \n\t"*
    "addpd     %xmm2,  %xmm8   \n\t"*  # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd 128(%rbx), %xmm2   \n\t"*  # tmp2     = _mm_load_pd(B+16)
    "addpd     %xmm6,  %xmm9   \n\t"*  # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"*  # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"*  # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"*  # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"*
    "addq      \$\$32*4,   %rbx    \n\t"*  # B += 16;
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm4,  %xmm10  \n\t"*  # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"*  # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"*  # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"*  # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd    (%rax), %xmm0   \n\t"*  # tmp0     = _mm_load_pd(A+16)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  16(%rax), %xmm1   \n\t"*  # tmp1     = _mm_load_pd(A+18)
    "                            \n\t"*
    "prefetcht2        0(%r10)  \n\t"*  # prefetch nextB[0]
    "prefetcht2       64(%r10)  \n\t"*  # prefetch nextB[8]
    "                            \n\t"*
    "decq      %rsi             \n\t"*  # --l
    "jne       .DLOOP\${:uid}          \n\t"*  # if l>= 1 go back
    "                            \n\t"*
    "                            \n\t"*
    ".DCONSIDERLEFT\${:uid}:           \n\t"*
    "testq     %rdi,   %rdi    \n\t"*  # if kl==0 writeback to AB
    "je        .DPOSTACCUMULATE\${:uid}\n\t"*
    "                            \n\t"*
    ".DLOOPLEFT\${:uid}:               \n\t"*  # for l = kl,..,1 do
    "                            \n\t"*
    "addpd     %xmm3,  %xmm12  \n\t"*  # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"*  # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"*  # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"*  # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"*  # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"*  # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm5,  %xmm14  \n\t"*  # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"*  # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"*  # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"*  # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm2,  %xmm8   \n\t"*  # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"*  # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"*  # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"*  # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"*  # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"*  #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"*  # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"*  # tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"*
    "                            \n\t"*
    "addpd     %xmm4,  %xmm10  \n\t"*  # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"*  # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"*  # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"*  # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"*  # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"*  # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"*  # tmp1     = _mm_load_pd(A+6)
    "                            \n\t"*
    "                            \n\t"*
    "addq      \$\$32,     %rax    \n\t"*  # A += 4;
    "addq      \$\$32,     %rbx    \n\t"*  # B += 4;
    "                            \n\t"*
    "decq      %rdi             \n\t"*  # --l
    "jne       .DLOOPLEFT\${:uid}      \n\t"*  # if l>= 1 go back
    "                            \n\t"*
    ".DPOSTACCUMULATE\${:uid}:         \n\t"*  # Update remaining ab_*_* registers
    "                            \n\t"*
    "addpd    %xmm3,   %xmm12  \n\t"*  # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "addpd    %xmm6,   %xmm13  \n\t"*  # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "                            \n\t"*  #
    "addpd    %xmm5,   %xmm14  \n\t"*  # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd    %xmm7,   %xmm15  \n\t"*  # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "                            \n\t"*
#
#  Update C <- beta*C + alpha*AB
#
#
    "movsd  \$4,                 %xmm0 \n\t"*  # load alpha
    "movsd  \$5,                 %xmm1 \n\t"*  # load beta
    "movq   \$6,                 %rcx  \n\t"*  # Address of C stored in %rcx

    "movq   \$7,                 %r8   \n\t"*  # load incRowC
    "leaq   (,%r8,8),           %r8   \n\t"*  #      incRowC *= sizeof(double)
    "movq   \$8,                 %r9   \n\t"*  # load incColC
    "leaq   (,%r9,8),           %r9   \n\t"*  #      incRowC *= sizeof(double)
    "                                   \n\t"*
    "leaq (%rcx,%r9),          %r10  \n\t"*  # Store addr of C01 in %r10
    "leaq (%rcx,%r8,2),        %rdx  \n\t"*  # Store addr of C20 in %rdx
    "leaq (%rdx,%r9),          %r11  \n\t"*  # Store addr of C21 in %r11
    "                                   \n\t"*
    "unpcklpd %xmm0,            %xmm0 \n\t"*  # duplicate alpha
    "unpcklpd %xmm1,            %xmm1 \n\t"*  # duplicate beta
    "                                   \n\t"*
    "                                   \n\t"*
    "movlpd (%rcx),             %xmm3 \n\t"*  # load (C00,
    "movhpd (%r10,%r8),        %xmm3 \n\t"*  #       C11)
    "mulpd  %xmm0,              %xmm8 \n\t"*  # scale ab_00_11 by alpha
    "mulpd  %xmm1,              %xmm3 \n\t"*  # scale (C00, C11) by beta
    "addpd  %xmm8,              %xmm3 \n\t"*  # add results

    "movlpd %xmm3,        (%rcx)       \n\t"*  # write back (C00,
    "movhpd %xmm3,        (%r10,%r8)  \n\t"*  #             C11)
    "                                   \n\t"*
    "movlpd (%rdx),             %xmm4 \n\t"*  # load (C20,
    "movhpd (%r11,%r8),        %xmm4 \n\t"*  #       C31)
    "mulpd  %xmm0,              %xmm9 \n\t"*  # scale ab_20_31 by alpha
    "mulpd  %xmm1,              %xmm4 \n\t"*  # scale (C20, C31) by beta
    "addpd  %xmm9,              %xmm4 \n\t"*  # add results
    "movlpd %xmm4,        (%rdx)       \n\t"*  # write back (C20,
    "movhpd %xmm4,        (%r11,%r8)  \n\t"*  #             C31)
    "                                   \n\t"*
    "                                   \n\t"*
    "movlpd (%r10),             %xmm3 \n\t"*  # load (C01,
    "movhpd (%rcx,%r8),        %xmm3 \n\t"*  #       C10)
    "mulpd  %xmm0,              %xmm10\n\t"*  # scale ab_01_10 by alpha
    "mulpd  %xmm1,              %xmm3 \n\t"*  # scale (C01, C10) by beta
    "addpd  %xmm10,             %xmm3 \n\t"*  # add results
    "movlpd %xmm3,        (%r10)      \n\t"*  # write back (C01,
    "movhpd %xmm3,        (%rcx,%r8) \n\t"*  #             C10)
    "                                   \n\t"*
    "movlpd (%r11),             %xmm4 \n\t"*  # load (C21,
    "movhpd (%rdx,%r8),        %xmm4 \n\t"*  #       C30)
    "mulpd  %xmm0,              %xmm11\n\t"*  # scale ab_21_30 by alpha
    "mulpd  %xmm1,              %xmm4 \n\t"*  # scale (C21, C30) by beta
    "addpd  %xmm11,             %xmm4 \n\t"*  # add results
    "movlpd %xmm4,        (%r11)      \n\t"*  # write back (C21,
    "movhpd %xmm4,        (%rdx,%r8) \n\t"*  #             C30)
    "                                   \n\t"*
    "                                   \n\t"*
    "leaq   (%rcx,%r9,2),      %rcx  \n\t"*  # Store addr of C02 in %rcx
    "leaq   (%r10,%r9,2),      %r10  \n\t"*  # Store addr of C03 in %r10
    "leaq   (%rdx,%r9,2),      %rdx  \n\t"*  # Store addr of C22 in $rdx
    "leaq   (%r11,%r9,2),      %r11  \n\t"*  # Store addr of C23 in %r11
    "                                   \n\t"*
    "                                   \n\t"*
    "movlpd (%rcx),             %xmm3 \n\t"*  # load (C02,
    "movhpd (%r10,%r8),        %xmm3 \n\t"*  #       C13)
    "mulpd  %xmm0,              %xmm12\n\t"*  # scale ab_02_13 by alpha
    "mulpd  %xmm1,              %xmm3 \n\t"*  # scale (C02, C13) by beta
    "addpd  %xmm12,             %xmm3 \n\t"*  # add results
    "movlpd %xmm3,        (%rcx)      \n\t"*  # write back (C02,
    "movhpd %xmm3,        (%r10,%r8) \n\t"*  #             C13)
    "                                   \n\t"*
    "movlpd (%rdx),             %xmm4 \n\t"*  # load (C22,
    "movhpd (%r11, %r8),       %xmm4 \n\t"*  #       C33)
    "mulpd  %xmm0,              %xmm13\n\t"*  # scale ab_22_33 by alpha
    "mulpd  %xmm1,              %xmm4 \n\t"*  # scale (C22, C33) by beta
    "addpd  %xmm13,             %xmm4 \n\t"*  # add results
    "movlpd %xmm4,             (%rdx) \n\t"*  # write back (C22,
    "movhpd %xmm4,        (%r11,%r8) \n\t"*  #             C33)
    "                                   \n\t"*
    "                                   \n\t"*
    "movlpd (%r10),             %xmm3 \n\t"*  # load (C03,
    "movhpd (%rcx,%r8),        %xmm3 \n\t"*  #       C12)
    "mulpd  %xmm0,              %xmm14\n\t"*  # scale ab_03_12 by alpha
    "mulpd  %xmm1,              %xmm3 \n\t"*  # scale (C03, C12) by beta
    "addpd  %xmm14,             %xmm3 \n\t"*  # add results
    "movlpd %xmm3,        (%r10)      \n\t"*  # write back (C03,
    "movhpd %xmm3,        (%rcx,%r8) \n\t"*  #             C12)
    "                                   \n\t"*
    "movlpd (%r11),             %xmm4 \n\t"*  # load (C23,
    "movhpd (%rdx,%r8),        %xmm4 \n\t"*  #       C32)
    "mulpd  %xmm0,              %xmm15\n\t"*  # scale ab_23_32 by alpha
    "mulpd  %xmm1,              %xmm4 \n\t"*  # scale (C23, C32) by beta
    "addpd  %xmm15,             %xmm4 \n\t"*  # add results
    "movlpd %xmm4,        (%r11)      \n\t"*  # write back (C23,
    "movhpd %xmm4,        (%rdx,%r8) \n\t"*  #             C32)
    "\",\""*
    # output
    # input
        "m,"* # (kb),       # 0
        "m,"* # (kl),       # 1
        "m,"* # (A),        # 2
        "m,"* # (B),        # 3
        "m,"* # (alpha),    # 4
        "m,"* # (beta),     # 5
        "m,"* # (C),        # 6
        "m,"* # (incRowC),  # 7
        "m,"* # (incColC),  # 8
        "m,"* # (nextA),    # 9
        "m,"* # (nextB)     # 10
    # register clobber list
        "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},"*
        "~{xmm0},~{xmm1},~{xmm2},~{xmm3},"*
        "~{xmm4},~{xmm5},~{xmm6},~{xmm7},"*
        "~{xmm8},~{xmm9},~{xmm10},~{xmm11},"*
        "~{xmm12},~{xmm13},~{xmm14},~{xmm15}\""*
        "(i64 %0, i64 %1, double* %2, double* %3, double %4, double %5, double* %6, i64 %7, i64 %8, double* %9, double* %10)"

@inline function gemm_micro_kernel(kc::Integer, α::Float64, A::Array{Float64}, Aoffset::Integer,
    B::Array{Float64}, Boffset::Integer, β::Float64, C::Array{Float64}, Coffset::Integer,
    incRowC::Integer, incColC::Integer, nextAoffset::Integer, nextBoffset::Integer)
    @inbounds begin
        ccall(:jl_breakpoint, Any, (), )
        kb, kl = divrem(kc, 4)

        Base.llvmcall("""
            call void asm $asms
            ret void""",
        Void,
        Tuple{Clong, Clong, Ptr{Float64}, Ptr{Float64}, Float64, Float64, Ptr{Float64}, Clong, Clong, Ptr{Float64}, Ptr{Float64}},
        Clong(kb), Clong(kl), pointer(A, Aoffset + 1), pointer(B, Boffset + 1), α, β, pointer(C, Coffset + 1), Clong(incRowC), Clong(incColC), pointer(A, nextAoffset + 1), pointer(B, nextBoffset + 1))
    end
end

function geaxpy(m::Integer, n::Integer, α::Float64, X::Array{Float64}, incRowX::Integer,
    incColX::Integer, Y::Array{Float64}, Yoffset::Integer, incRowY::Integer, incColY::Integer)

    if α != 1.0
        for j = 1:n
            for i = 1:m
                Y[Yoffset + (i - 1)*incRowY + (j - 1)*incColY + 1] = Y[(i - 1)*incRowY + (j - 1)*incColY + 1] + α*X[(i - 1)*incRowX + (j - 1)*incColX + 1]
            end
        end
    else
        for j = 1:n
            for i = 1:m
                Y[Yoffset + (i - 1)*incRowY + (j - 1)*incColY + 1] = Y[(i - 1)*incRowY + (j - 1)*incColY + 1] + X[(i - 1)*incRowX + (j - 1)*incColX + 1]
            end
        end
    end
end

function gescal(m::Integer,
                n::Integer,
                α::Float64,
                X::Array{Float64},
                Xoffset::Integer,
                incRowX::Integer,
                incColX::Integer)

    if α != 0.0
        for j = 1:n
            for i = 1:m
                X[Xoffset + (i - 1)*incRowX + (j - 1)*incColX + 1] *= α
            end
        end
    else
        for j = 1:n
            for i = 1:m
                X[Xoffset + (i - 1)*incRowX + (j - 1)*incColX + 1] = 0.0
            end
        end
    end
end

function gemm_macro_kernel(mc::Integer, nc::Integer, kc::Integer, α::Float64,
    β::Float64, C::Array{Float64}, Coffset::Integer, incRowC::Integer, incColC::Integer)

    mp = div(mc + MR - 1, MR)
    np = div(nc + NR - 1, NR)

    _mr = mc % MR
    _nr = nc % NR

    for j = 1:np
        nr = (j != np || _nr == 0) ? NR : _nr
        nextBoffset = (j - 1)*kc*NR

        for i = 1:mp
            mr = (i != mp || _mr == 0) ? MR : _mr
            nextAoffset = i*kc*MR

            if i == mp
                nextAoffset = (i - 1)*kc*MR
                nextBoffset = j*kc*NR
                if j == np
                    nextBoffset = (j - 1)*kc*NR
                end
            end

            if mr == MR && nr == NR
                gemm_micro_kernel(kc,
                    α,
                    _A,
                    (i - 1)*kc*MR,
                    _B,
                    (j - 1)*kc*NR,
                    β,
                    C,
                    Coffset + (i - 1)*MR*incRowC + (j - 1)*NR*incColC,
                    incRowC,
                    incColC,
                    nextAoffset,
                    nextBoffset)
            else
                gemm_micro_kernel(kc,
                    α,
                    _A,
                    (i - 1)*kc*MR,
                    _B,
                    (j - 1)*kc*NR,
                    0.0,
                    C,
                    Coffset,
                    1,
                    MR,
                    nextAoffset,
                    nextBoffset)
                gescal(mr, nr, β, C, Coffset + (i - 1)*MR*incRowC + (j - 1)*NR*incColC, incRowC, incColC)
                geaxpy(mr, nr, 1.0, _C, 1, MR, C, Coffset + (i - 1)*MR*incRowC + (j - 1)*NR*incColC, incRowC, incColC)
            end
        end
    end
    return C
end

function gemm_nn(m::Integer,
                 n::Integer,
                 k::Integer,
                 α::Float64,
                 A::Matrix{Float64},
                 incRowA::Integer,
                 incColA::Integer,
                 B::Matrix{Float64},
                 incRowB::Integer,
                 incColB::Integer,
                 β::Float64,
                 C::Matrix{Float64},
                 incRowC::Integer,
                 incColC::Integer)

    mb = div(m + MC - 1, MC)
    nb = div(n + NC - 1, NC)
    kb = div(k + KC - 1, KC)

    _mc = m % MC
    _nc = n % NC
    _kc = k % KC

    if α == 0.0 || k == 0
        gescal(m, n, β, C, incRowC, incColC)
        return C
    end

    for j = 1:nb
        nc = (j != nb || _nc == 0) ? NC : _nc

        for l = 1:kb
            kc = (l != kb || _kc == 0) ? KC : _kc
            _β = l == 1 ? β : 1.0

            pack_B(kc, nc, B, (l - 1)*KC*incRowB + (j - 1)*NC*incColB, incRowB, incColB, _B)

            for i = 1:mb
                mc = (i != mb || _mc == 0) ? MC : _mc

                pack_A(mc, kc, A, (i - 1)*MC*incRowA + (l - 1)*KC*incColA, incRowA, incColA, _A)

                gemm_macro_kernel(mc, nc, kc, α, _β, C, (i - 1)*MC*incRowC + (j - 1)*NC*incColC, incRowC, incColC)
            end
        end
    end
    return C
end

end
