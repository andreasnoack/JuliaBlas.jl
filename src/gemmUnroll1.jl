module LC

const MC = 384
const KC = 384
const NC = 4096

const MR = 4
const NR = 4

const _A = Array(Float64, MC*KC)
const _B = Array(Float64, KC*NC)
const _C = Array(Float64, MR*NR)

const _AB = Array(Float64, MR*NR)
const pAB = pointer(_AB)

p_A = pointer(_A)
p_B = pointer(_B)
p_C = pointer(_C)

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
  "\"movl      \$0,      %esi    \n\t"* # kb (32 bit) stored in %esi
    "movl      \$1,      %edi    \n\t"* # kl (32 bit) stored in %edi
    "movq      \$2,      %rax    \n\t"* # Address of A stored in %rax
    "movq      \$3,      %rbx    \n\t"* # Address of B stored in %rbx
    "movq      \$4,      %rcx    \n\t"* # Address of AB stored in %rcx

    "movapd    (%rax), %xmm0   \n\t"* # tmp0 = _mm_load_pd(A)
    "movapd  16(%rax), %xmm1   \n\t"* # tmp1 = _mm_load_pd(A+2)
    "movapd    (%rbx), %xmm2   \n\t"* # tmp2 = _mm_load_pd(B)

    "xorpd     %xmm8,  %xmm8   \n\t"* # ab_00_11 = _mm_setzero_pd()
    "xorpd     %xmm9,  %xmm9   \n\t"* # ab_20_31 = _mm_setzero_pd()
    "xorpd     %xmm10, %xmm10  \n\t"* # ab_01_10 = _mm_setzero_pd()
    "xorpd     %xmm11, %xmm11  \n\t"* # ab_21_30 = _mm_setzero_pd()
    "xorpd     %xmm12, %xmm12  \n\t"* # ab_02_13 = _mm_setzero_pd()
    "xorpd     %xmm13, %xmm13  \n\t"* # ab_22_33 = _mm_setzero_pd()
    "xorpd     %xmm14, %xmm14  \n\t"* # ab_03_12 = _mm_setzero_pd()
    "xorpd     %xmm15, %xmm15  \n\t"* # ab_23_32 = _mm_setzero_pd()

    "xorpd     %xmm3,  %xmm3   \n\t"* # tmp3 = _mm_setzero_pd
    "xorpd     %xmm4,  %xmm4   \n\t"* # tmp4 = _mm_setzero_pd
    "xorpd     %xmm5,  %xmm5   \n\t"* # tmp5 = _mm_setzero_pd
    "xorpd     %xmm6,  %xmm6   \n\t"* # tmp6 = _mm_setzero_pd
    "xorpd     %xmm7,  %xmm7   \n\t"* # tmp7 = _mm_setzero_pd
    "testl     %edi,   %edi    \n\t"* # if kl==0 writeback to AB


    "testl     %esi,   %esi    \n\t"* # if kb==0 handle remaining kl
    "je        .DCONSIDERLEFT  \n\t"* # update iterations

    ".DLOOP:                   \n\t"* # for l = kb,..,1 do

                                      # 1. update
    "addpd     %xmm3,  %xmm12  \n\t"* # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"* # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"* # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"* # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"* # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"* # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1);


    "addpd     %xmm5,  %xmm14  \n\t"* # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"* # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"* # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"* # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)


    "addpd     %xmm2,  %xmm8   \n\t"* # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"* # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"* # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"* # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"* # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"* # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1)


    "addpd     %xmm4,  %xmm10  \n\t"* # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"* # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"* # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"* # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"* # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"* # tmp1     = _mm_load_pd(A+6)


    "addq      \$\$32,     %rax    \n\t"* # A += 4;
    "addq      \$\$32,     %rbx    \n\t"* # B += 4;

                                        # 2. update
    "addpd     %xmm3,  %xmm12  \n\t"* # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"* # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"* # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"* # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"* # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"* # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1);


    "addpd     %xmm5,  %xmm14  \n\t"* # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"* # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"* # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"* # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)


    "addpd     %xmm2,  %xmm8   \n\t"* # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"* # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"* # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"* # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"* # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
                                       #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"* # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1)


    "addpd     %xmm4,  %xmm10  \n\t"* # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"* # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"* # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"* # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"* # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"* # tmp1     = _mm_load_pd(A+6)


    "addq      \$\$32,     %rax    \n\t"* # A += 4;
    "addq      \$\$32,     %rbx    \n\t"* # B += 4;

                                        # 3. update
    "addpd     %xmm3,  %xmm12  \n\t"* # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"* # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"* # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"* # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"* # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
                                       #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"* # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1);


    "addpd     %xmm5,  %xmm14  \n\t"* # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"* # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"* # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"* # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)


    "addpd     %xmm2,  %xmm8   \n\t"* # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"* # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"* # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"* # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"* # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"* # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1)


    "addpd     %xmm4,  %xmm10  \n\t"* # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"* # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"* # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"* # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"* # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"* # tmp1     = _mm_load_pd(A+6)


    "addq      \$\$32,     %rax    \n\t"* # A += 4;
    "addq      \$\$32,     %rbx    \n\t"* # B += 4;

                                      # 4. update
    "addpd     %xmm3,  %xmm12  \n\t"* # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"* # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"* # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"* # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"* # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"* # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1);


    "addpd     %xmm5,  %xmm14  \n\t"* # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"* # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"* # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"* # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)


    "addpd     %xmm2,  %xmm8   \n\t"* # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"* # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"* # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"* # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"* # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"* # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1)


    "addpd     %xmm4,  %xmm10  \n\t"* # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"* # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"* # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"* # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"* # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"* # tmp1     = _mm_load_pd(A+6)


    "addq      \$\$32,     %rax    \n\t"* # A += 4;
    "addq      \$\$32,     %rbx    \n\t"* # B += 4;

    "decl      %esi             \n\t"* # --l
    "jne       .DLOOP          \n\t"* # if l>= 1 go back


    ".DCONSIDERLEFT:           \n\t"*
    "testl     %edi,   %edi    \n\t"* # if kl==0 writeback to AB
    "je        .DPOSTACCUMULATE    \n\t"*

    ".DLOOPLEFT:               \n\t"* # for l = kl,..,1 do

    "addpd     %xmm3,  %xmm12  \n\t"* # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd  16(%rbx), %xmm3   \n\t"* # tmp3     = _mm_load_pd(B+2)
    "addpd     %xmm6,  %xmm13  \n\t"* # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %xmm2,  %xmm6   \n\t"* # tmp6     = tmp2
    "pshufd \$\$78,%xmm2, %xmm4   \n\t"* # tmp4     = _mm_shuffle_pd(tmp2, tmp2,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm2   \n\t"* # tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1);


    "addpd     %xmm5,  %xmm14  \n\t"* # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd     %xmm7,  %xmm15  \n\t"* # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %xmm4,  %xmm7   \n\t"* # tmp7     = tmp4
    "mulpd     %xmm0,  %xmm4   \n\t"* # tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)


    "addpd     %xmm2,  %xmm8   \n\t"* # ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd  32(%rbx), %xmm2   \n\t"* # tmp2     = _mm_load_pd(B+4)
    "addpd     %xmm6,  %xmm9   \n\t"* # ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %xmm3,  %xmm6   \n\t"* # tmp6     = tmp3
    "pshufd \$\$78,%xmm3, %xmm5   \n\t"* # tmp5     = _mm_shuffle_pd(tmp3, tmp3,
                                      #                   _MM_SHUFFLE2(0, 1))
    "mulpd     %xmm0,  %xmm3   \n\t"* # tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %xmm1,  %xmm6   \n\t"* # tmp6     = _mm_mul_pd(tmp6, tmp1)


    "addpd     %xmm4,  %xmm10  \n\t"* # ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "addpd     %xmm7,  %xmm11  \n\t"* # ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %xmm5,  %xmm7   \n\t"* # tmp7     = tmp5
    "mulpd     %xmm0,  %xmm5   \n\t"* # tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd  32(%rax), %xmm0   \n\t"* # tmp0     = _mm_load_pd(A+4)
    "mulpd     %xmm1,  %xmm7   \n\t"* # tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd  48(%rax), %xmm1   \n\t"* # tmp1     = _mm_load_pd(A+6)


    "addq      \$\$32,     %rax    \n\t"* # A += 4;
    "addq      \$\$32,     %rbx    \n\t"* # B += 4;

    "decl      %edi             \n\t"* # --l
    "jne       .DLOOPLEFT      \n\t"* # if l>= 1 go back

    ".DPOSTACCUMULATE:         \n\t"* # Update remaining ab_*_* registers

    "addpd    %xmm3,   %xmm12  \n\t"* # ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "addpd    %xmm6,   %xmm13  \n\t"* # ab_22_33 = _mm_add_pd(ab_22_33, tmp6)

    "addpd    %xmm5,   %xmm14  \n\t"* # ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "addpd    %xmm7,   %xmm15  \n\t"* # ab_23_32 = _mm_add_pd(ab_23_32, tmp7)

    ".DWRITEBACK:              \n\t"* # Fill AB with computed values

                                      # Fill first col of AB:
    "movlpd   %xmm8,    (%rcx) \n\t"* # _mm_storel_pd(&AB[0+0*4], ab_00_11)
    "movhpd   %xmm10 , 8(%rcx) \n\t"* # _mm_storeh_pd(&AB[1+0*4], ab_01_10)
    "movlpd   %xmm9,  16(%rcx) \n\t"* # _mm_storel_pd(&AB[2+0*4], ab_20_31)
    "movhpd   %xmm11, 24(%rcx) \n\t"* # _mm_storeh_pd(&AB[3+0*4], ab_21_30)

    "addq     \$\$32,       %rcx   \n\t"* # Fill second col of AB:
    "movlpd   %xmm10,   (%rcx) \n\t"* # _mm_storel_pd(&AB[0+1*4], ab_01_10)
    "movhpd   %xmm8,   8(%rcx) \n\t"* # _mm_storeh_pd(&AB[1+1*4], ab_00_11)
    "movlpd   %xmm11, 16(%rcx) \n\t"* # _mm_storel_pd(&AB[2+1*4], ab_21_30)
    "movhpd   %xmm9,  24(%rcx) \n\t"* # _mm_storeh_pd(&AB[3+1*4], ab_20_31)

    "addq     \$\$32,       %rcx   \n\t"* # Fill third col of AB:
    "movlpd   %xmm12,   (%rcx) \n\t"* # _mm_storel_pd(&AB[0+2*4], ab_02_13)
    "movhpd   %xmm14,  8(%rcx) \n\t"* # _mm_storeh_pd(&AB[1+2*4], ab_03_12)
    "movlpd   %xmm13, 16(%rcx) \n\t"* # _mm_storel_pd(&AB[2+2*4], ab_22_33)
    "movhpd   %xmm15, 24(%rcx) \n\t"* # _mm_storeh_pd(&AB[3+2*4], ab_23_32)

    "addq     \$\$32,       %rcx   \n\t"* # Fill forth col of AB:
    "movlpd   %xmm14,   (%rcx) \n\t"* # _mm_storel_pd(&AB[0+3*4], ab_03_12)
    "movhpd   %xmm12,  8(%rcx) \n\t"* # _mm_storeh_pd(&AB[1+3*4], ab_02_13)
    "movlpd   %xmm15, 16(%rcx) \n\t"* # _mm_storel_pd(&AB[2+3*4], ab_23_32)
    "movhpd   %xmm13, 24(%rcx) \n\t"* # _mm_storeh_pd(&AB[3+3*4], ab_22_33)
    "\",\""*
    # output
    # input
    "r,"* #(kb),     # 0
    "r,"* #(kl),     # 1
    "r,"* #(A),      # 2
    "r,"* #(B),      # 3
    "r,"* #(AB)      # 4
    # register clobber list
    "~{rax},~{rbx},~{rcx},~{esi},~{edi},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}\"(i32 %0, i32 %1, double* %2, double* %3, double* %4)"

@inline function gemm_micro_kernel(kc::Integer, α::Float64, A::Array{Float64}, Aoffset::Integer,
    B::Array{Float64}, Boffset::Integer, β::Float64, C::Array{Float64}, Coffset::Integer,
    incRowC::Integer, incColC::Integer)
    @inbounds begin

        kb, kl = divrem(kc, 4)

        Base.llvmcall("""
            call void asm $asms
            ret void""",
        Void,
        Tuple{Cint, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}},
        Cint(kb), Cint(kl), pointer(A, Aoffset + 1), pointer(B, Boffset + 1), pointer(_AB))


#  Update C <- beta*C
    if β == 0.0
        for j = 1:NR
            for i =1:MR
                C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] = 0.0
            end
        end
    elseif β != 1.0
        for j = 1:NR
            for i = 1:MR
                C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] *= β
            end
        end
    end

#  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
#                                  the above layer dgemm_nn)
    if α == 1.0
        for j = 1:NR
            for i = 1:MR
                C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] += _AB[i + (j - 1)*MR]
            end
        end
    else
        for j = 1:NR
            for i = 1:MR
                C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] += α*_AB[i + (j - 1)*MR]
            end
        end
    end
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

        for i = 1:mp
            mr = (i != mp || _mr == 0) ? MR : _mr

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
                    incColC)
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
                    MR)
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
