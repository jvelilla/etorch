note
	description: "Tests for ET_LAYER_NORM."

class
	ET_TEST_LAYER_NORM

inherit
	EQA_TEST_SET

feature -- Tests

	test_layer_norm_forward
		local
			ln: ET_LAYER_NORM
			x, y: ET_TENSOR
			x_store: ET_STORAGE_REAL_64
			tol: REAL_64
		do
			tol := 1.0e-5
			
			-- Input x: shape [2, 3]
			create x_store.make (6)
			x_store.put_real_64 (1.0, 1)
			x_store.put_real_64 (2.0, 2)
			x_store.put_real_64 (3.0, 3)
			x_store.put_real_64 (4.0, 4)
			x_store.put_real_64 (5.0, 5)
			x_store.put_real_64 (6.0, 6)
			
			create x.make_from_storage (x_store, <<2, 3>>, <<3, 1>>, 0)
			
			-- LayerNorm over last dim (normalized_shape=3)
			create ln.make (3, 1.0e-5)
			
			-- Forward pass
			y := ln.forward (x)
			
			assert ("Result shape count", y.shape.count = 2)
			assert ("Result shape M", y.shape [1] = 2)
			assert ("Result shape N", y.shape [2] = 3)
			
			-- Expected output for row 1 [1, 2, 3]:
			-- mean = 2, var = 1 (unbiased=False)
			-- normalized = [(1-2)/1, (2-2)/1, (3-2)/1] = [-1, 0, 1]
			if attached {ET_STORAGE_REAL_64} y.storage as ys then
				assert_approx_32 (ys.item_as_real_64 (1), -1.224735, tol, "y[1,1]")
				assert_approx_32 (ys.item_as_real_64 (2), 0.0, tol, "y[1,2]")
				assert_approx_32 (ys.item_as_real_64 (3), 1.224735, tol, "y[1,3]")
				-- Row 2 [4, 5, 6]: mean = 5, var = 1, norm = [-1, 0, 1]
				assert_approx_32 (ys.item_as_real_64 (4), -1.224735, tol, "y[2,1]")
				assert_approx_32 (ys.item_as_real_64 (5), 0.0, tol, "y[2,2]")
				assert_approx_32 (ys.item_as_real_64 (6), 1.224735, tol, "y[2,3]")
			else
				assert ("Storage should be real 64", False)
			end
		end

feature {NONE} -- Helpers

	assert_approx_32 (actual, expected, tol: REAL_64; msg: STRING)
		do
			assert (msg + " Expected " + expected.out + " but got " + actual.out, (actual - expected).abs <= tol)
		end

end
