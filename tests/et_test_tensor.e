note
	description: "Tests for ET_TENSOR basic ops and Matmul."

class
	ET_TEST_TENSOR

inherit
	EQA_TEST_SET

feature -- Tests

	test_tensor_creation
		local
			x: ET_TENSOR
			l_shape: ARRAY [INTEGER_32]
		do
			l_shape := <<3>>
			create x.make_zeros (l_shape)

			assert ("Correct shape", x.numel = 3)
		end

	test_tensor_matmul
		local
			a, b, c: ET_TENSOR
			a_store, b_store: ET_STORAGE_REAL_64
			l_strides: ARRAY [INTEGER_32]
			tol: REAL_64
		do
			tol := 1.0e-5
			
			create a_store.make (4)
			a_store.put_real_64 (1.0, 1)
			a_store.put_real_64 (2.0, 2)
			a_store.put_real_64 (3.0, 3)
			a_store.put_real_64 (4.0, 4)
			
			create b_store.make (4)
			b_store.put_real_64 (2.0, 1)
			b_store.put_real_64 (0.0, 2)
			b_store.put_real_64 (1.0, 3)
			b_store.put_real_64 (2.0, 4)
			
			l_strides := <<2, 1>>
			
			create a.make_from_storage (a_store, <<2, 2>>, l_strides, 0)
			create b.make_from_storage (b_store, <<2, 2>>, l_strides, 0)
			
			c := a.matmul (b)
			assert ("MATMUL Result count", c.shape.count = 2)
			assert ("MATMUL Result shape M", c.shape [1] = 2)
			assert ("MATMUL Result shape N", c.shape [2] = 2)
			
			if attached {ET_STORAGE_REAL_64} c.storage as cs then
				assert_approx_32 (cs.item_as_real_64 (1), 4.0, tol, "C[1,1]")
				assert_approx_32 (cs.item_as_real_64 (2), 4.0, tol, "C[1,2]")
				assert_approx_32 (cs.item_as_real_64 (3), 10.0, tol, "C[2,1]")
				assert_approx_32 (cs.item_as_real_64 (4), 8.0, tol, "C[2,2]")
			else
				assert ("Storage should be real 64", False)
			end
		end

	test_plus_in_place
		local
			a, b: ET_TENSOR
			a_store, b_store: ET_STORAGE_REAL_64
			l_strides: ARRAY [INTEGER_32]
			tol: REAL_64
			orig_store: ET_STORAGE
		do
			tol := 1.0e-5
			
			create a_store.make (4)
			a_store.put_real_64 (1.0, 1)
			a_store.put_real_64 (2.0, 2)
			a_store.put_real_64 (3.0, 3)
			a_store.put_real_64 (4.0, 4)
			
			create b_store.make (4)
			b_store.put_real_64 (2.0, 1)
			b_store.put_real_64 (0.0, 2)
			b_store.put_real_64 (1.0, 3)
			b_store.put_real_64 (2.0, 4)
			
			l_strides := <<2, 1>>
			
			create a.make_from_storage (a_store, <<2, 2>>, l_strides, 0)
			create b.make_from_storage (b_store, <<2, 2>>, l_strides, 0)
			
			orig_store := a.storage
			a.plus_in_place (b)
			
			assert ("Storage instance is identical", a.storage = orig_store)
			
			if attached {ET_STORAGE_REAL_64} a.storage as as_store then
				assert_approx_32 (as_store.item_as_real_64 (1), 3.0, tol, "A[1,1]")
				assert_approx_32 (as_store.item_as_real_64 (2), 2.0, tol, "A[1,2]")
				assert_approx_32 (as_store.item_as_real_64 (3), 4.0, tol, "A[2,1]")
				assert_approx_32 (as_store.item_as_real_64 (4), 6.0, tol, "A[2,2]")
			else
				assert ("Storage should be real 64", False)
			end
		end

	test_matmul_1d_vectors
		local
			a, b, c: ET_TENSOR
			a_store, b_store: ET_STORAGE_REAL_64
			tol: REAL_64
		do
			tol := 1.0e-5
			create a_store.make (3)
			a_store.put_real_64 (1.0, 1)
			a_store.put_real_64 (2.0, 2)
			a_store.put_real_64 (3.0, 3)
			
			create b_store.make (3)
			b_store.put_real_64 (4.0, 1)
			b_store.put_real_64 (5.0, 2)
			b_store.put_real_64 (6.0, 3)
			
			create a.make_from_storage (a_store, <<3>>, <<1>>, 0)
			create b.make_from_storage (b_store, <<3>>, <<1>>, 0)
			
			c := a.matmul (b)
			assert ("1D dot Result count", c.shape.count = 0)
			
			if attached {ET_STORAGE_REAL_64} c.storage as cs then
				assert_approx_32 (cs.item_as_real_64 (1), 32.0, tol, "Dot")
			else
				assert ("Storage should be real 64", False)
			end
		end

	test_sum_multi_dim
		local
			a, c: ET_TENSOR
			a_store: ET_STORAGE_REAL_64
			tol: REAL_64
		do
			tol := 1.0e-5
			create a_store.make (8)
			a_store.put_real_64 (1.0, 1)
			a_store.put_real_64 (2.0, 2)
			a_store.put_real_64 (3.0, 3)
			a_store.put_real_64 (4.0, 4)
			a_store.put_real_64 (5.0, 5)
			a_store.put_real_64 (6.0, 6)
			a_store.put_real_64 (7.0, 7)
			a_store.put_real_64 (8.0, 8)
			
			create a.make_from_storage (a_store, <<2, 2, 2>>, <<4, 2, 1>>, 0)
			
			c := a.sum (<<1, 2>>, False)
			
			assert ("Sum shape count", c.shape.count = 1)
			assert ("Sum shape[1]", c.shape [1] = 2)
			
			if attached {ET_STORAGE_REAL_64} c.storage as cs then
				assert_approx_32 (cs.item_as_real_64 (1), 16.0, tol, "Sum[0]")
				assert_approx_32 (cs.item_as_real_64 (2), 20.0, tol, "Sum[1]")
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
