note
    description: "Tests for KV Cache in Multi-Head Attention."

class
    ET_TEST_KV_CACHE

inherit
    EQA_TEST_SET
        redefine
            on_prepare
        select
			default_create
        end
    DOUBLE_MATH
        rename
            default_create as dm_default_create
        end

feature -- Initialization

    on_prepare
        do
        end

feature -- Tests

    test_kv_cache_logic
        local
            mha: ET_MULTIHEAD_ATTENTION
            n_embd, n_head, block_size: INTEGER
            dropout: REAL_64

            x1, x2, x_full: ET_TENSOR
            out1, out2_1, out2_2: ET_TENSOR

            tol: REAL_64
        do
            print ("  [TEST] KV Cache... ")
            n_embd := 4
            n_head := 2
            block_size := 10
            dropout := 0.0
            tol := 1.0e-5

            create mha.make (n_embd, n_head, dropout)

            -- Create input sequence [B=1, T=2, C=4]
            create x_full.make_zeros (<<1, 2, 4>>)

            -- 1. Run without cache (Standard)
            out1 := mha.forward (x_full)

            -- 2. Run with cache (Step by Step)
            mha.init_cache (10)

            -- Step 1: Input 1 [1, 1, 4]
            create x1.make_zeros (<<1, 1, 4>>)

            out2_1 := mha.forward (x1)

            -- Step 2: Input 2 [1, 1, 4]
            create x2.make_zeros (<<1, 1, 4>>)

            out2_2 := mha.forward (x2)

            check_approx (out2_2, out1.slice_range (2, 2, 1), tol)

            print ("OK%N")
        end

    check_approx (t1, t2: ET_TENSOR; tol: REAL_64)
        local
            i: INTEGER
            v1, v2: REAL_64
        do
            assert ("Shape match", t1.shape.count = t2.shape.count)
            from i := 1 until i > t1.numel loop
                v1 := t1.storage.item_as_real_64 (t1.offset + i)
                v2 := t2.storage.item_as_real_64 (t2.offset + i)
                assert ("Value mismatch at " + i.out + ": " + v1.out + " vs " + v2.out, (v1 - v2).abs <= tol)
                i := i + 1
            end
        end

end
