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

feature -- DType Tests

	test_zeros_real_32
			-- Verify a REAL_32 tensor is created with the correct storage.
		local
			t: ET_TENSOR
			l_dtype: ET_DTYPE
		do
			create {ET_DTYPE_FLOAT32} l_dtype
			create t.make_zeros_with_dtype (<<3>>, l_dtype)
			assert ("numel is 3", t.numel = 3)
			assert ("dtype is float32", t.dtype.is_float32)
			assert ("storage is ET_STORAGE_REAL_32", attached {ET_STORAGE_REAL_32} t.storage)
		end

	test_zeros_int_32
			-- Verify an INTEGER_32 tensor is created with the correct storage.
		local
			t: ET_TENSOR
			l_dtype: ET_DTYPE
		do
			create {ET_DTYPE_INT32} l_dtype
			create t.make_zeros_with_dtype (<<4>>, l_dtype)
			assert ("numel is 4", t.numel = 4)
			assert ("dtype is int32", t.dtype.is_int32)
			assert ("storage is ET_STORAGE_INT_32", attached {ET_STORAGE_INT_32} t.storage)
		end

	test_zeros_bool
			-- Verify a BOOLEAN tensor is created with the correct storage.
		local
			t: ET_TENSOR
			l_dtype: ET_DTYPE
		do
			create {ET_DTYPE_BOOL} l_dtype
			create t.make_zeros_with_dtype (<<2>>, l_dtype)
			assert ("numel is 2", t.numel = 2)
			assert ("dtype is bool", t.dtype.is_bool)
			assert ("storage is ET_STORAGE_BOOL", attached {ET_STORAGE_BOOL} t.storage)
		end

	test_plus_real_32
			-- Verify element-wise addition works on REAL_32 tensors.
		local
			a, b, c: ET_TENSOR
			l_store_a, l_store_b: ET_STORAGE_REAL_32
			tol: REAL_64
		do
			tol := 1.0e-5
			create l_store_a.make (3)
			l_store_a.put_real_32 (1.0, 1)
			l_store_a.put_real_32 (2.0, 2)
			l_store_a.put_real_32 (3.0, 3)

			create l_store_b.make (3)
			l_store_b.put_real_32 (10.0, 1)
			l_store_b.put_real_32 (20.0, 2)
			l_store_b.put_real_32 (30.0, 3)

			create a.make_from_storage (l_store_a, <<3>>, <<1>>, 0)
			create b.make_from_storage (l_store_b, <<3>>, <<1>>, 0)
			-- Make both float32
			a.set_dtype (create {ET_DTYPE_FLOAT32})
			b.set_dtype (create {ET_DTYPE_FLOAT32})

			c := a + b

			assert ("Result dtype is float32", c.dtype.is_float32)
			assert ("Result storage is REAL_32", attached {ET_STORAGE_REAL_32} c.storage)
			if attached {ET_STORAGE_REAL_32} c.storage as cs then
				assert_approx_32 (cs.item_as_real_32 (1), 11.0, tol, "c[1]")
				assert_approx_32 (cs.item_as_real_32 (2), 22.0, tol, "c[2]")
				assert_approx_32 (cs.item_as_real_32 (3), 33.0, tol, "c[3]")
			end
		end

	test_plus_int_32
			-- Verify element-wise addition works on INTEGER_32 tensors.
		local
			a, b, c: ET_TENSOR
			l_store_a, l_store_b: ET_STORAGE_INT_32
		do
			create l_store_a.make (3)
			l_store_a.put_int_32 (1, 1)
			l_store_a.put_int_32 (2, 2)
			l_store_a.put_int_32 (3, 3)

			create l_store_b.make (3)
			l_store_b.put_int_32 (10, 1)
			l_store_b.put_int_32 (20, 2)
			l_store_b.put_int_32 (30, 3)

			create a.make_from_storage (l_store_a, <<3>>, <<1>>, 0)
			create b.make_from_storage (l_store_b, <<3>>, <<1>>, 0)
			a.set_dtype (create {ET_DTYPE_INT32})
			b.set_dtype (create {ET_DTYPE_INT32})

			c := a + b

			assert ("Result dtype is int32", c.dtype.is_int32)
			assert ("Result storage is INT_32", attached {ET_STORAGE_INT_32} c.storage)
			if attached {ET_STORAGE_INT_32} c.storage as cs then
				assert ("c[1] = 11", cs.item_as_int_32 (1) = 11)
				assert ("c[2] = 22", cs.item_as_int_32 (2) = 22)
				assert ("c[3] = 33", cs.item_as_int_32 (3) = 33)
			end
		end

	test_mul_bool_and
			-- Verify element-wise multiplication on BOOLEAN tensors performs logical AND.
		local
			a, b, c: ET_TENSOR
			l_store_a, l_store_b: ET_STORAGE_BOOL
		do
			create l_store_a.make (3)
			l_store_a.put_boolean (True, 1)
			l_store_a.put_boolean (True, 2)
			l_store_a.put_boolean (False, 3)

			create l_store_b.make (3)
			l_store_b.put_boolean (True, 1)
			l_store_b.put_boolean (False, 2)
			l_store_b.put_boolean (False, 3)

			create a.make_from_storage (l_store_a, <<3>>, <<1>>, 0)
			create b.make_from_storage (l_store_b, <<3>>, <<1>>, 0)
			a.set_dtype (create {ET_DTYPE_BOOL})
			b.set_dtype (create {ET_DTYPE_BOOL})

			c := a * b

			assert ("Result dtype is bool", c.dtype.is_bool)
			assert ("Result storage is BOOL", attached {ET_STORAGE_BOOL} c.storage)
			if attached {ET_STORAGE_BOOL} c.storage as cs then
				assert ("c[1] = True AND True = True", cs.item_as_boolean (1) = True)
				assert ("c[2] = True AND False = False", cs.item_as_boolean (2) = False)
				assert ("c[3] = False AND False = False", cs.item_as_boolean (3) = False)
			end
		end

	test_sum_dim_int_32
			-- Verify sum reduction works on INTEGER_32 tensors.
		local
			t, s: ET_TENSOR
			l_store: ET_STORAGE_INT_32
		do
			-- Tensor [1, 2, 3, 4] reshaped as shape [2,2]
			create l_store.make (4)
			l_store.put_int_32 (1, 1)
			l_store.put_int_32 (2, 2)
			l_store.put_int_32 (3, 3)
			l_store.put_int_32 (4, 4)
			create t.make_from_storage (l_store, <<2, 2>>, <<2, 1>>, 0)
			t.set_dtype (create {ET_DTYPE_INT32})

			s := t.sum (<<2>>, False)

			assert ("Sum shape is [2]", s.shape.count = 1 and s.shape [1] = 2)
			assert ("Sum dtype is int32", s.dtype.is_int32)
			if attached {ET_STORAGE_INT_32} s.storage as cs then
				assert ("s[1] = 3 (1+2)", cs.item_as_int_32 (1) = 3)
				assert ("s[2] = 7 (3+4)", cs.item_as_int_32 (2) = 7)
			else
				assert ("Storage should be INTEGER_32", False)
			end
		end

	test_matmul_float32
			-- Verify 2x2 matmul works via BLAS sgemm for float32 tensors.
		local
			a, b, c: ET_TENSOR
			a_store, b_store: ET_STORAGE_REAL_32
			l_strides: ARRAY [INTEGER_32]
			tol: REAL_64
		do
			tol := 1.0e-3
			-- A = [[1, 2], [3, 4]]
			create a_store.make (4)
			a_store.put_real_32 (1.0, 1)
			a_store.put_real_32 (2.0, 2)
			a_store.put_real_32 (3.0, 3)
			a_store.put_real_32 (4.0, 4)
			-- B = [[2, 0], [1, 2]]
			create b_store.make (4)
			b_store.put_real_32 (2.0, 1)
			b_store.put_real_32 (0.0, 2)
			b_store.put_real_32 (1.0, 3)
			b_store.put_real_32 (2.0, 4)

			l_strides := <<2, 1>>
			create a.make_from_storage (a_store, <<2, 2>>, l_strides, 0)
			create b.make_from_storage (b_store, <<2, 2>>, l_strides, 0)
			a.set_dtype (create {ET_DTYPE_FLOAT32})
			b.set_dtype (create {ET_DTYPE_FLOAT32})

			c := a.matmul (b)
			assert ("F32 MATMUL Result shape", c.shape [1] = 2 and c.shape [2] = 2)
			assert ("F32 MATMUL dtype is float32", c.dtype.is_float32)
			assert ("F32 MATMUL storage is REAL_32", attached {ET_STORAGE_REAL_32} c.storage)
			if attached {ET_STORAGE_REAL_32} c.storage as cs then
				assert_approx_32 (cs.item_as_real_32 (1), 4.0, tol, "C[1,1]")
				assert_approx_32 (cs.item_as_real_32 (2), 4.0, tol, "C[1,2]")
				assert_approx_32 (cs.item_as_real_32 (3), 10.0, tol, "C[2,1]")
				assert_approx_32 (cs.item_as_real_32 (4), 8.0, tol, "C[2,2]")
			end
		end

	test_matmul_int32
			-- Verify 2x2 matmul works via generic fallback for int32 tensors.
		local
			a, b, c: ET_TENSOR
			a_store, b_store: ET_STORAGE_INT_32
			l_strides: ARRAY [INTEGER_32]
		do
			-- A = [[1, 2], [3, 4]]
			create a_store.make (4)
			a_store.put_int_32 (1, 1)
			a_store.put_int_32 (2, 2)
			a_store.put_int_32 (3, 3)
			a_store.put_int_32 (4, 4)
			-- B = [[2, 0], [1, 2]]
			create b_store.make (4)
			b_store.put_int_32 (2, 1)
			b_store.put_int_32 (0, 2)
			b_store.put_int_32 (1, 3)
			b_store.put_int_32 (2, 4)

			l_strides := <<2, 1>>
			create a.make_from_storage (a_store, <<2, 2>>, l_strides, 0)
			create b.make_from_storage (b_store, <<2, 2>>, l_strides, 0)
			a.set_dtype (create {ET_DTYPE_INT32})
			b.set_dtype (create {ET_DTYPE_INT32})

			c := a.matmul (b)
			assert ("I32 MATMUL Result shape", c.shape [1] = 2 and c.shape [2] = 2)
			assert ("I32 MATMUL dtype is int32", c.dtype.is_int32)
			-- Generic fallback produces INT_64 storage for integer dtypes
			if attached {ET_STORAGE_INT_64} c.storage as cs then
				assert ("C[1,1] = 4", cs.item_as_int_64 (1) = 4)
				assert ("C[1,2] = 4", cs.item_as_int_64 (2) = 4)
				assert ("C[2,1] = 10", cs.item_as_int_64 (3) = 10)
				assert ("C[2,2] = 8", cs.item_as_int_64 (4) = 8)
			else
				assert ("Storage should be INT_64 (generic fallback for int32 input)", False)
			end
		end

end
