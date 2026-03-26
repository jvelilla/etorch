note
	description: "[
		N-dimensional Tensor backed by `ET_STORAGE`.
		Features strong Design by Contract constraints for mathematical correctness.
	]"

class
	ET_TENSOR

inherit
	ANY
		redefine
			out
		end

create
	make_from_storage,
	make_zeros,
	make_ones,
	make_randn,
	make_zeros_with_dtype,
	make_ones_with_dtype,
	make_randn_with_dtype,
	make_from_array2d

feature {NONE} -- Initialization

	make_from_storage (a_storage: ET_STORAGE; a_shape: ARRAY [INTEGER_32]; a_strides: ARRAY [INTEGER_32]; a_offset: INTEGER_32)
			-- Create a tensor view over an existing storage.
		require
			shape_lower_one: a_shape.lower = 1
			strides_match_shape: a_strides.count = a_shape.count
			offset_valid: a_offset >= 0
		do
			storage := a_storage
			shape := a_shape.deep_twin
			strides := a_strides.deep_twin
			offset := a_offset

			create device.make_cpu
			if attached {ET_STORAGE_REAL_64} a_storage then
				create {ET_DTYPE_FLOAT64} dtype
			elseif attached {ET_STORAGE_REAL_32} a_storage then
				create {ET_DTYPE_FLOAT32} dtype
			elseif attached {ET_STORAGE_INT_32} a_storage then
				create {ET_DTYPE_INT32} dtype
			elseif attached {ET_STORAGE_INT_64} a_storage then
				create {ET_DTYPE_INT64} dtype
			elseif attached {ET_STORAGE_BOOL} a_storage then
				create {ET_DTYPE_BOOL} dtype
			else
				dtype := {ET_DTYPE_REGISTRY}.default_float_dtype
			end
		ensure
			storage_set: storage = a_storage
			offset_set: offset = a_offset
		end

	make_zeros (a_shape: ARRAY [INTEGER_32])
			-- Create a new ZERO-initialized tensor of default float dtype.
		require
			shape_lower_one: a_shape.lower = 1
		do
			make_zeros_with_dtype (a_shape, {ET_DTYPE_REGISTRY}.default_float_dtype)
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_ones (a_shape: ARRAY [INTEGER_32])
			-- Create a new ONE-initialized tensor of default float dtype.
		require
			shape_lower_one: a_shape.lower = 1
		do
			make_ones_with_dtype (a_shape, {ET_DTYPE_REGISTRY}.default_float_dtype)
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_randn (a_shape: ARRAY [INTEGER_32])
			-- Create a new normal-distributed random tensor of default float dtype.
		require
			shape_lower_one: a_shape.lower = 1
		do
			make_randn_with_dtype (a_shape, {ET_DTYPE_REGISTRY}.default_float_dtype)
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_zeros_with_dtype (a_shape: ARRAY [INTEGER_32]; a_dtype: ET_DTYPE)
			-- Create a new ZERO-initialized tensor of specific dtype.
		require
			shape_lower_one: a_shape.lower = 1
			valid_dtype: a_dtype /= Void
		local
			l_count: INTEGER_32
			l_strides: ARRAY [INTEGER_32]
			l_factory: ET_STORAGE_FACTORY
		do
			shape := a_shape.deep_twin
			l_strides := calculate_contiguous_strides (a_shape)
			strides := l_strides
			offset := 0
			l_count := calculate_product (a_shape)
			create l_factory
			storage := l_factory.make_zeros (l_count, a_dtype)
			create device.make_cpu
			dtype := a_dtype
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_ones_with_dtype (a_shape: ARRAY [INTEGER_32]; a_dtype: ET_DTYPE)
			-- Create a new ONE-initialized tensor of specific dtype.
		require
			shape_lower_one: a_shape.lower = 1
			valid_dtype: a_dtype /= Void
		local
			l_count: INTEGER_32
			l_strides: ARRAY [INTEGER_32]
			l_factory: ET_STORAGE_FACTORY
		do
			shape := a_shape.deep_twin
			l_strides := calculate_contiguous_strides (a_shape)
			strides := l_strides
			offset := 0
			l_count := calculate_product (a_shape)
			create l_factory
			storage := l_factory.make_ones (l_count, a_dtype)
			create device.make_cpu
			dtype := a_dtype
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_randn_with_dtype (a_shape: ARRAY [INTEGER_32]; a_dtype: ET_DTYPE)
			-- Create a new random normal-distributed tensor of specific dtype.
		require
			shape_lower_one: a_shape.lower = 1
			valid_dtype: a_dtype /= Void
			valid_floating_type: a_dtype.is_floating
		local
			l_count: INTEGER_32
			l_strides: ARRAY [INTEGER_32]
			l_factory: ET_STORAGE_FACTORY
		do
			shape := a_shape.deep_twin
			l_strides := calculate_contiguous_strides (a_shape)
			strides := l_strides
			offset := 0
			l_count := calculate_product (a_shape)
			create l_factory
			storage := l_factory.make_randn (l_count, a_dtype)
			create device.make_cpu
			dtype := a_dtype
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_from_array2d (values: ARRAY [ARRAY [REAL_64]])
			-- Create a tensor from a 2D array literal.
		require
			valid_rows: values.count > 0
			valid_cols: values [values.lower].count > 0
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_32
			i, j, m, n: INTEGER_32
			l_idx: INTEGER_32
		do
			m := values.count
			n := values [values.lower].count
			shape := <<m, n>>
			l_strides := calculate_contiguous_strides (shape)
			strides := l_strides
			offset := 0

			create l_store.make (m * n)
			storage := l_store
			from i := 1 until i > m loop
				from j := 1 until j > n loop
					l_idx := (i - 1) * n + (j - 1)
					l_store.put_real_32 (values [values.lower + i - 1] [values [values.lower + i - 1].lower + j - 1].truncated_to_real, l_idx + 1)
					j := j + 1
				end
				i := i + 1
			end
			create device.make_cpu
			create {ET_DTYPE_FLOAT32} dtype
		ensure
			shape_set: shape [1] = values.count and shape [2] = values [values.lower].count
		end

feature -- Element Access

	scalar_value: REAL_64
			-- Return the single scalar value of this tensor.
			-- Equivalent to `.item()` in PyTorch.
		require
			is_scalar: rank = 0 or calculate_product (shape) = 1
		local
			zero_offset: INTEGER_32
		do
			zero_offset := 0
			if attached {ET_STORAGE_REAL_64} storage as target_store then
				Result := target_store.item_as_real_64 (offset + zero_offset + 1)
			elseif attached {ET_STORAGE_REAL_32} storage as target_store then
				Result := target_store.item_as_real_32 (offset + zero_offset + 1)
			else
				(create {EXCEPTIONS}).raise ("Storage is not REAL_64 or REAL_32 for scalar_value")
			end
		end

	to_array: ARRAY [REAL_64]
			-- Flatten storage and return as a raw Eiffel array.
			-- Equivalent to `.numpy().flatten()`.
		local
			i, total: INTEGER_32
		do
			total := calculate_product (shape)
			create Result.make_empty
			if attached {ET_STORAGE_REAL_64} storage as s then
				from i := 1 until i > total loop
					Result.force (s.item_as_real_64 (offset + i), i)
					i := i + 1
				end
			elseif attached {ET_STORAGE_REAL_32} storage as s then
				from i := 1 until i > total loop
					Result.force (s.item_as_real_32 (offset + i), i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("Storage is not REAL_64 or REAL_32 for to_array")
			end
		end

	put_real_64 (v: REAL_64; a_indices: ARRAY [INTEGER_32])
			-- Set the element at N-D index `a_indices` to `v`.
		require
			valid_indices: a_indices.count = shape.count
		local
			i, flat_idx: INTEGER_32
		do
			flat_idx := 0
			from i := a_indices.upper until i < a_indices.lower loop
				flat_idx := flat_idx + (a_indices [i] - 1) * strides [i]
				i := i - 1
			end
			if attached {ET_STORAGE_REAL_64} storage as target_store then
				target_store.put_real_64 (v, offset + flat_idx + 1)
			elseif attached {ET_STORAGE_REAL_32} storage as target_store then
				target_store.put_real_32 (v.truncated_to_real, offset + flat_idx + 1)
			else
				(create {EXCEPTIONS}).raise ("Storage is not REAL_64 or REAL_32")
			end
		end

	item_as_real_64 (a_indices: ARRAY [INTEGER_32]): REAL_64
			-- Get the element at N-D index `a_indices`.
		require
			valid_indices: a_indices.count = shape.count
		local
			i, flat_idx: INTEGER_32
		do
			flat_idx := 0
			from i := a_indices.upper until i < a_indices.lower loop
				flat_idx := flat_idx + (a_indices [i] - 1) * strides [i]
				i := i - 1
			end
			if attached {ET_STORAGE_REAL_64} storage as target_store then
				Result := target_store.item_as_real_64 (offset + flat_idx + 1)
			elseif attached {ET_STORAGE_REAL_32} storage as target_store then
				Result := target_store.item_as_real_32 (offset + flat_idx + 1)
			else
				(create {EXCEPTIONS}).raise ("Storage is not REAL_64 or REAL_32")
			end
		end

feature -- Properties

	shape: ARRAY [INTEGER_32]
			-- Dimensions of the tensor.

	strides: ARRAY [INTEGER_32]
			-- Step sizes for each dimension.

	storage: ET_STORAGE
			-- Underlying flat data.

	offset: INTEGER_32
			-- Starting index offset in the storage (0-based mathematically, adapt internally to 1-based Eiffel structs).

	device: ET_DEVICE
			-- Where is this tensor located (CPU, CUDA, etc.)

	dtype: ET_DTYPE
			-- Data type of the tensor.

	requires_grad: BOOLEAN
			-- Autograd tracking.

	rank: INTEGER_32
			-- Number of dimensions (e.g. 2 for matrix).
		do
			Result := shape.count
		end

	grad_node: detachable ET_VALUE
			-- Autograd graph node.

	set_grad_node (a_node: ET_VALUE)
			-- Link this tensor to a graph node.
		do
			grad_node := a_node
		end

	ensure_grad_node: ET_VALUE
			-- Return existing grad_node or create a new leaf node.
		do
			if attached grad_node as n then
				Result := n
			else
				create Result.make (Current)
				grad_node := Result
			end
		ensure
			node_exists: grad_node /= Void
			result_consistent: Result = grad_node
		end

	numel: INTEGER_32
			-- Total number of elements physically representing this shape.
		do
			Result := calculate_product (shape)
		end

feature -- Autograd Props

	grad: detachable ET_TENSOR
			-- Gradient of this tensor.

	set_grad (a_grad: detachable ET_TENSOR)
			-- Manually set the gradient (used dynamically or for tests).
		do
			grad := a_grad
		end

	set_requires_grad (val: BOOLEAN)
			-- Enable/disable Autograd tracking.
		do
			requires_grad := val
		end

feature {ANY, ET_TENSOR, ET_VALUE, ET_MATMUL_FUNCTION, ET_ADD_FUNCTION, ET_MUL_FUNCTION}

	set_dtype (a_dtype: ET_DTYPE)
			-- Set the data type of this tensor.
		do
			dtype := a_dtype
		end

feature -- Autograd Props (continued)

	backward
			-- Trigger backpropagation.
		require
			requires_grad: requires_grad
		do
			if attached grad_node as node then
				if not attached grad then
					-- Root gradient is 1.0 of the same shape/dtype
					set_grad (Current.make_ones_with_dtype_like)
				end
				if attached grad as g then
					node.set_grad (g)
				end
				node.backward
			else
				-- If no node, but requires_grad is true, it's a leaf we just track.
				-- Backward from a leaf with no operations is a no-op or just sets its own grad to 1.
				if not attached grad then
					set_grad (Current.make_ones_with_dtype_like)
				end
			end
		end

	make_ones_with_dtype_like: ET_TENSOR
			-- Creates a tensor of ones matching shape and dtype of Current
		do
			create Result.make_ones_with_dtype (shape, dtype)
		end

feature -- Math Operations (with strict Contracts)

	minus alias "-" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise subtraction.
		require
			same_shape: is_broadcastable (other.shape)

		local
			v1, v2: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_ADD_FUNCTION
			minus_other: ET_TENSOR
		do
			if {ET_TORCH}.is_grad_enabled and then (requires_grad or other.requires_grad) then
				minus_other := other.mul_scalar (-1.0)
				v1 := ensure_grad_node
				v2 := minus_other.ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1, v2>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := plus_internal (other.mul_scalar (-1.0))
			end
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	plus alias "+" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise addition.
		require
			same_shape: is_broadcastable (other.shape)

		local
			v1, v2: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_ADD_FUNCTION
		do
			if {ET_TORCH}.is_grad_enabled and then (requires_grad or other.requires_grad) then
				v1 := ensure_grad_node
				v2 := other.ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1, v2>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := plus_internal (other)
			end
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	plus_internal (other: ET_TENSOR): ET_TENSOR
			-- Numeric implementation of addition.
		local
			l_strides, br_shape: ARRAY [INTEGER_32]
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_store_i32: ET_STORAGE_INT_32
			l_count, i, idx_self, idx_other: INTEGER_32
			l_promoter: ET_DTYPE_PROMOTER
			l_target_dtype: ET_DTYPE
			l_self, l_other: ET_TENSOR
		do
			create l_promoter
			l_target_dtype := l_promoter.promoted_dtype (dtype, other.dtype)
			l_self := if dtype.is_equal (l_target_dtype) then Current else to_dtype (l_target_dtype) end
			l_other := if other.dtype.is_equal (l_target_dtype) then other else other.to_dtype (l_target_dtype) end

			br_shape := broadcast_shape (l_self.shape, l_other.shape)
			l_count := calculate_product (br_shape)
			l_strides := calculate_contiguous_strides (br_shape)

			if l_target_dtype.is_float64 then
				create l_store_f64.make (l_count)
				l_store := l_store_f64
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_f64.put_real_64 (l_self.storage.item_as_real_64 (l_self.offset + idx_self + 1) + l_other.storage.item_as_real_64 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			elseif l_target_dtype.is_float32 then
				create l_store_f32.make (l_count)
				l_store := l_store_f32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_f32.put_real_32 (l_self.storage.item_as_real_32 (l_self.offset + idx_self + 1) + l_other.storage.item_as_real_32 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			elseif l_target_dtype.is_int32 then
				create l_store_i32.make (l_count)
				l_store := l_store_i32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_i32.put_int_32 (l_self.storage.item_as_int_32 (l_self.offset + idx_self + 1) + l_other.storage.item_as_int_32 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for addition")
				create l_store_f64.make (0)
				l_store := l_store_f64
			end

			create Result.make_from_storage (l_store, br_shape, l_strides, 0)
			Result.set_dtype (l_target_dtype)
		end

	plus_in_place (other: ET_TENSOR)
			-- In-place element-wise addition.
		require
			not_requires_grad: not requires_grad
			same_shape: is_broadcastable (other.shape)

		local
			br_shape: ARRAY [INTEGER_32]
			l_count, i, idx_self, idx_other: INTEGER_32
		do
			br_shape := broadcast_shape (shape, other.shape)
			l_count := calculate_product (br_shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_real_64 (s.item_as_real_64 (offset + idx_self + 1) + other.storage.item_as_real_64 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_real_32 (s.item_as_real_32 (offset + idx_self + 1) + other.storage.item_as_real_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_int32 then
				if attached {ET_STORAGE_INT_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_int_32 (s.item_as_int_32 (offset + idx_self + 1) + other.storage.item_as_int_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_bool then
				if attached {ET_STORAGE_BOOL} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_boolean (s.item_as_boolean (offset + idx_self + 1) or other.storage.item_as_boolean (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for plus_in_place")
			end
		end

	plus_scalar (val: REAL_64): ET_TENSOR
			-- Element-wise addition with a scalar.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			l_strides := calculate_contiguous_strides (shape)

			if dtype.is_float64 then
				create l_store_f64.make (l_count)
				l_store := l_store_f64
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, shape, strides, shape)
					l_store_f64.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) + val, i)
					i := i + 1
				end
			elseif dtype.is_float32 then
				create l_store_f32.make (l_count)
				l_store := l_store_f32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, shape, strides, shape)
					l_store_f32.put_real_32 (storage.item_as_real_32 (offset + idx_self + 1) + val.truncated_to_real, i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("plus_scalar only supports float tensors currently")
				create l_store_f64.make (0)
				l_store := l_store_f64
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
			Result.set_dtype (dtype)
		end

	plus_scalar_in_place (val: REAL_64)
			-- In-place element-wise addition with a scalar.
		require
			not_requires_grad: not requires_grad
		local
			l_count, i, idx_self: INTEGER_32

		do
			l_count := calculate_product (shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						s.put_real_64 (s.item_as_real_64 (offset + idx_self + 1) + val, offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						s.put_real_32 (s.item_as_real_32 (offset + idx_self + 1) + val.truncated_to_real, offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				(create {EXCEPTIONS}).raise ("plus_scalar_in_place only supports float tensors currently")
			end
		end

	matmul (other: ET_TENSOR): ET_TENSOR
			-- Matrix multiplication (supports 1D dot, 2D matmul, and N-D batched matmul).
			-- Dispatches to BLAS fast paths for float types, generic fallback for others.
		require
			valid_operands: rank >= 1 and other.rank >= 1

		local
			v1, v2: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_MATMUL_FUNCTION
		do
			if {ET_TORCH}.is_grad_enabled and then (requires_grad or other.requires_grad) then
				v1 := ensure_grad_node
				v2 := other.ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1, v2>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := matmul_internal (other)
			end
		end

	matmul_internal (other: ET_TENSOR): ET_TENSOR
			-- Numeric implementation of matmul.
		local
			l_res: ET_TENSOR
			l_new_shape: ARRAY [INTEGER_32]
			i: INTEGER_32
			l_promoter: ET_DTYPE_PROMOTER
			l_target_dtype: ET_DTYPE
			l_self, l_other: ET_TENSOR
		do
			if rank = 1 and other.rank >= 2 then
				create l_new_shape.make_filled (1, 1, 2)
				l_new_shape[1] := 1
				l_new_shape[2] := shape[1]
				l_res := reshape (l_new_shape).matmul_internal (other)

				create l_new_shape.make_empty
				from i := 1 until i > l_res.rank loop
					if i /= l_res.rank - 1 then
						l_new_shape.force (l_res.shape[i], l_new_shape.count + 1)
					end
					i := i + 1
				end
				Result := l_res.reshape (l_new_shape)
			else
				create l_promoter
				l_target_dtype := l_promoter.promoted_dtype (dtype, other.dtype)
				l_self := if dtype.is_equal (l_target_dtype) then Current else to_dtype (l_target_dtype) end
				l_other := if other.dtype.is_equal (l_target_dtype) then other else other.to_dtype (l_target_dtype) end

				if l_target_dtype.is_float64 then
					Result := l_self.matmul_blas_f64 (l_other)
				elseif l_target_dtype.is_float32 then
					Result := l_self.matmul_blas_f32 (l_other)
				else
					-- Generic fallback for int64, int32, bool (no BLAS equivalent)
					Result := l_self.matmul_generic (l_other)
				end
				Result.set_dtype (l_target_dtype)
			end
		end

feature -- Matmul Helpers (BLAS fast paths + generic fallback)

	matmul_blas_f64 (other: ET_TENSOR): ET_TENSOR
			-- Matrix multiplication via BLAS cblas_dgemm (float64).
		local
			l_res_shape: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_64
			l_res_strides: ARRAY [INTEGER_32]
			l_m, l_n, l_k: INTEGER_32
			l_lda, l_ldb, l_ldc: INTEGER_32
			l_trans_a, l_trans_b: INTEGER_32
			l_blas: ET_BLAS
			i, l_iter: INTEGER_32
			a_batch_shape, b_batch_shape, res_batch_shape: ARRAY [INTEGER_32]
			l_batch_count: INTEGER_32
			offset_c: INTEGER_32
			idx_a_batch, idx_b_batch: INTEGER_32
			l_dot: REAL_64
		do
			if rank = 1 and other.rank = 1 then
				-- 1D dot product
				l_k := shape [1]
				create l_res_shape.make_empty
				create l_res_strides.make_empty
				create l_res_store.make (1)
				l_dot := 0.0
				from i := 0 until i >= l_k loop
					l_dot := l_dot + storage.item_as_real_64 (offset + i * strides [1] + 1) *
					                 other.storage.item_as_real_64 (other.offset + i * other.strides [1] + 1)
					i := i + 1
				end
				l_res_store.put_real_64 (l_dot, 1)
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)

			elseif rank >= 2 and other.rank = 1 then
				-- Matrix-Vector product
				create l_res_shape.make_empty
				from i := 1 until i > rank - 1 loop
					l_res_shape.force (shape [i], l_res_shape.count + 1)
					i := i + 1
				end
				l_res_strides := calculate_contiguous_strides (l_res_shape)
				l_batch_count := calculate_product (l_res_shape)
				create l_res_store.make (l_batch_count)
				l_k := other.shape[1]
				from i := 1 until i > l_batch_count loop
					idx_a_batch := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_dot := 0.0
					from l_iter := 0 until l_iter >= l_k loop
						l_dot := l_dot + storage.item_as_real_64(offset + idx_a_batch + l_iter * strides[rank] + 1) * other.storage.item_as_real_64(other.offset + l_iter * other.strides[1] + 1)
						l_iter := l_iter + 1
					end
					l_res_store.put_real_64(l_dot, i)
					i := i + 1
				end
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)

			else
				-- N-D Batched Matmul / 2D Matmul via BLAS dgemm
				create a_batch_shape.make_empty
				from i := 1 until i > rank - 2 loop
					a_batch_shape.force (shape [i], a_batch_shape.count + 1)
					i := i + 1
				end
				create b_batch_shape.make_empty
				from i := 1 until i > other.rank - 2 loop
					b_batch_shape.force (other.shape [i], b_batch_shape.count + 1)
					i := i + 1
				end
				res_batch_shape := broadcast_shape(a_batch_shape, b_batch_shape)
				l_m := shape [rank - 1]
				l_k := shape [rank]
				l_n := other.shape [other.rank]
				l_res_shape := res_batch_shape.deep_twin
				l_res_shape.force(l_m, l_res_shape.count + 1)
				l_res_shape.force(l_n, l_res_shape.count + 1)
				l_batch_count := calculate_product(res_batch_shape)
				if l_batch_count = 0 then l_batch_count := 1 end
				create l_res_store.make (l_batch_count * l_m * l_n)
				l_res_strides := calculate_contiguous_strides (l_res_shape)
				create l_blas
				if strides.count >= 2 and then strides [rank-1] = 1 and then strides [rank] = l_m then
					l_trans_a := 112
					l_lda := l_m
				else
					l_trans_a := 111
					l_lda := l_k
				end
				if other.strides.count >= 2 and then other.strides [other.rank-1] = 1 and then other.strides [other.rank] = l_k then
					l_trans_b := 112
					l_ldb := l_k
				else
					l_trans_b := 111
					l_ldb := l_n
				end
				l_ldc := l_n
				if res_batch_shape.count = 0 then
					l_blas.cblas_dgemm (101, l_trans_a, l_trans_b, l_m, l_n, l_k, 1.0,
						storage.data_pointer + offset * dtype.byte_size, l_lda,
						other.storage.data_pointer + other.offset * dtype.byte_size, l_ldb,
						0.0, l_res_store.data_pointer, l_ldc)
				else
					from i := 1 until i > l_batch_count loop
						idx_a_batch := linear_index_to_offset(i, a_batch_shape, strides, res_batch_shape)
						idx_b_batch := linear_index_to_offset(i, b_batch_shape, other.strides, res_batch_shape)
						offset_c := (i - 1) * l_m * l_n
						l_blas.cblas_dgemm (101, l_trans_a, l_trans_b, l_m, l_n, l_k, 1.0,
							storage.data_pointer + (offset + idx_a_batch) * dtype.byte_size, l_lda,
							other.storage.data_pointer + (other.offset + idx_b_batch) * dtype.byte_size, l_ldb,
							0.0, l_res_store.data_pointer + offset_c * dtype.byte_size, l_ldc)
						i := i + 1
					end
				end
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)
			end
		end

	matmul_blas_f32 (other: ET_TENSOR): ET_TENSOR
			-- Matrix multiplication via BLAS cblas_sgemm (float32).
		local
			l_res_shape: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_32
			l_res_strides: ARRAY [INTEGER_32]
			l_m, l_n, l_k: INTEGER_32
			l_lda, l_ldb, l_ldc: INTEGER_32
			l_trans_a, l_trans_b: INTEGER_32
			l_blas: ET_BLAS
			i, l_iter: INTEGER_32
			a_batch_shape, b_batch_shape, res_batch_shape: ARRAY [INTEGER_32]
			l_batch_count: INTEGER_32
			offset_c: INTEGER_32
			idx_a_batch, idx_b_batch: INTEGER_32
			l_dot: REAL_32
		do
			if rank = 1 and other.rank = 1 then
				-- 1D dot product
				l_k := shape [1]
				create l_res_shape.make_empty
				create l_res_strides.make_empty
				create l_res_store.make (1)
				l_dot := {REAL_32} 0.0
				from i := 0 until i >= l_k loop
					l_dot := l_dot + storage.item_as_real_32 (offset + i * strides [1] + 1) *
					                 other.storage.item_as_real_32 (other.offset + i * other.strides [1] + 1)
					i := i + 1
				end
				l_res_store.put_real_32 (l_dot, 1)
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)

			elseif rank >= 2 and other.rank = 1 then
				-- Matrix-Vector product
				create l_res_shape.make_empty
				from i := 1 until i > rank - 1 loop
					l_res_shape.force (shape [i], l_res_shape.count + 1)
					i := i + 1
				end
				l_res_strides := calculate_contiguous_strides (l_res_shape)
				l_batch_count := calculate_product (l_res_shape)
				create l_res_store.make (l_batch_count)
				l_k := other.shape[1]
				from i := 1 until i > l_batch_count loop
					idx_a_batch := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_dot := {REAL_32} 0.0
					from l_iter := 0 until l_iter >= l_k loop
						l_dot := l_dot + storage.item_as_real_32(offset + idx_a_batch + l_iter * strides[rank] + 1) * other.storage.item_as_real_32(other.offset + l_iter * other.strides[1] + 1)
						l_iter := l_iter + 1
					end
					l_res_store.put_real_32(l_dot, i)
					i := i + 1
				end
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)

			else
				-- N-D Batched Matmul / 2D Matmul via BLAS sgemm
				create a_batch_shape.make_empty
				from i := 1 until i > rank - 2 loop
					a_batch_shape.force (shape [i], a_batch_shape.count + 1)
					i := i + 1
				end
				create b_batch_shape.make_empty
				from i := 1 until i > other.rank - 2 loop
					b_batch_shape.force (other.shape [i], b_batch_shape.count + 1)
					i := i + 1
				end
				res_batch_shape := broadcast_shape(a_batch_shape, b_batch_shape)
				l_m := shape [rank - 1]
				l_k := shape [rank]
				l_n := other.shape [other.rank]
				l_res_shape := res_batch_shape.deep_twin
				l_res_shape.force(l_m, l_res_shape.count + 1)
				l_res_shape.force(l_n, l_res_shape.count + 1)
				l_batch_count := calculate_product(res_batch_shape)
				if l_batch_count = 0 then l_batch_count := 1 end
				create l_res_store.make (l_batch_count * l_m * l_n)
				l_res_strides := calculate_contiguous_strides (l_res_shape)
				create l_blas
				if strides.count >= 2 and then strides [rank-1] = 1 and then strides [rank] = l_m then
					l_trans_a := 112
					l_lda := l_m
				else
					l_trans_a := 111
					l_lda := l_k
				end
				if other.strides.count >= 2 and then other.strides [other.rank-1] = 1 and then other.strides [other.rank] = l_k then
					l_trans_b := 112
					l_ldb := l_k
				else
					l_trans_b := 111
					l_ldb := l_n
				end
				l_ldc := l_n
				if res_batch_shape.count = 0 then
					l_blas.cblas_sgemm (101, l_trans_a, l_trans_b, l_m, l_n, l_k, {REAL_32} 1.0,
						storage.data_pointer + offset * dtype.byte_size, l_lda,
						other.storage.data_pointer + other.offset * dtype.byte_size, l_ldb,
						{REAL_32} 0.0, l_res_store.data_pointer, l_ldc)
				else
					from i := 1 until i > l_batch_count loop
						idx_a_batch := linear_index_to_offset(i, a_batch_shape, strides, res_batch_shape)
						idx_b_batch := linear_index_to_offset(i, b_batch_shape, other.strides, res_batch_shape)
						offset_c := (i - 1) * l_m * l_n
						l_blas.cblas_sgemm (101, l_trans_a, l_trans_b, l_m, l_n, l_k, {REAL_32} 1.0,
							storage.data_pointer + (offset + idx_a_batch) * dtype.byte_size, l_lda,
							other.storage.data_pointer + (other.offset + idx_b_batch) * dtype.byte_size, l_ldb,
							{REAL_32} 0.0, l_res_store.data_pointer + offset_c * dtype.byte_size, l_ldc)
						i := i + 1
					end
				end
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)
			end
		end

	matmul_generic (other: ET_TENSOR): ET_TENSOR
		-- Generic matrix multiplication via pure Eiffel loops.
		-- For int32/int64 dtypes: accumulates in INTEGER_64 and stores in ET_STORAGE_INT_64,
		-- ensuring dtype and storage type always agree.
		-- For bool dtype: accumulates in REAL_64 and stores in ET_STORAGE_REAL_64.
		local
			l_res_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_res_store_real: ET_STORAGE_REAL_64
			l_res_store_int: ET_STORAGE_INT_64
			l_m, l_n, l_k: INTEGER_32
			i, j, p, b_idx: INTEGER_32
			a_batch_shape, b_batch_shape, res_batch_shape: ARRAY [INTEGER_32]
			l_batch_count: INTEGER_32
			idx_a_batch: INTEGER_32
			l_dot_real: REAL_64
			l_dot_int: INTEGER_64
			l_a_int, l_b_int: INTEGER_64
			l_a_val, l_b_val: REAL_64
			l_use_int: BOOLEAN
		do
			l_use_int := dtype.is_int32 or dtype.is_int64

			if rank = 1 and other.rank = 1 then
				-- 1D dot product
				l_k := shape [1]
				create l_res_shape.make_empty
				create l_res_strides.make_empty
				if l_use_int then
					create l_res_store_int.make (1)
					l_dot_int := 0
					from i := 0 until i >= l_k loop
						l_a_int := storage_item_as_int_64_universal (storage, offset + i * strides [1] + 1)
						l_b_int := storage_item_as_int_64_universal (other.storage, other.offset + i * other.strides [1] + 1)
						l_dot_int := l_dot_int + l_a_int * l_b_int
						i := i + 1
					end
					l_res_store_int.put_int_64 (l_dot_int, 1)
					create Result.make_from_storage (l_res_store_int, l_res_shape, l_res_strides, 0)
				else
					-- bool: accumulate as real
					create l_res_store_real.make (1)
					l_dot_real := 0.0
					from i := 0 until i >= l_k loop
						l_a_val := storage_item_as_real_64_universal (storage, offset + i * strides [1] + 1)
						l_b_val := storage_item_as_real_64_universal (other.storage, other.offset + i * other.strides [1] + 1)
						l_dot_real := l_dot_real + l_a_val * l_b_val
						i := i + 1
					end
					l_res_store_real.put_real_64 (l_dot_real, 1)
					create Result.make_from_storage (l_res_store_real, l_res_shape, l_res_strides, 0)
				end

			elseif rank >= 2 and other.rank = 1 then
				-- Matrix-Vector product
				create l_res_shape.make_empty
				from i := 1 until i > rank - 1 loop
					l_res_shape.force (shape [i], l_res_shape.count + 1)
					i := i + 1
				end
				l_res_strides := calculate_contiguous_strides (l_res_shape)
				l_batch_count := calculate_product (l_res_shape)
				l_k := other.shape[1]
				if l_use_int then
					create l_res_store_int.make (l_batch_count)
					from i := 1 until i > l_batch_count loop
						idx_a_batch := linear_index_to_offset (i, shape, strides, l_res_shape)
						l_dot_int := 0
						from j := 0 until j >= l_k loop
							l_a_int := storage_item_as_int_64_universal (storage, offset + idx_a_batch + j * strides[rank] + 1)
							l_b_int := storage_item_as_int_64_universal (other.storage, other.offset + j * other.strides[1] + 1)
							l_dot_int := l_dot_int + l_a_int * l_b_int
							j := j + 1
						end
						l_res_store_int.put_int_64 (l_dot_int, i)
						i := i + 1
					end
					create Result.make_from_storage (l_res_store_int, l_res_shape, l_res_strides, 0)
				else
					create l_res_store_real.make (l_batch_count)
					from i := 1 until i > l_batch_count loop
						idx_a_batch := linear_index_to_offset (i, shape, strides, l_res_shape)
						l_dot_real := 0.0
						from j := 0 until j >= l_k loop
							l_a_val := storage_item_as_real_64_universal (storage, offset + idx_a_batch + j * strides[rank] + 1)
							l_b_val := storage_item_as_real_64_universal (other.storage, other.offset + j * other.strides[1] + 1)
							l_dot_real := l_dot_real + l_a_val * l_b_val
							j := j + 1
						end
						l_res_store_real.put_real_64(l_dot_real, i)
						i := i + 1
					end
					create Result.make_from_storage (l_res_store_real, l_res_shape, l_res_strides, 0)
				end

			else
				-- N-D Batched Matmul / 2D Matmul (pure Eiffel loops)
				create a_batch_shape.make_empty
				from i := 1 until i > rank - 2 loop
					a_batch_shape.force (shape [i], a_batch_shape.count + 1)
					i := i + 1
				end
				create b_batch_shape.make_empty
				from i := 1 until i > other.rank - 2 loop
					b_batch_shape.force (other.shape [i], b_batch_shape.count + 1)
					i := i + 1
				end
				res_batch_shape := broadcast_shape(a_batch_shape, b_batch_shape)
				l_m := shape [rank - 1]
				l_k := shape [rank]
				l_n := other.shape [other.rank]
				l_res_shape := res_batch_shape.deep_twin
				l_res_shape.force(l_m, l_res_shape.count + 1)
				l_res_shape.force(l_n, l_res_shape.count + 1)
				l_batch_count := calculate_product(res_batch_shape)
				if l_batch_count = 0 then l_batch_count := 1 end
				l_res_strides := calculate_contiguous_strides (l_res_shape)

				-- Triple-loop matmul for each batch
				if l_use_int then
					create l_res_store_int.make (l_batch_count * l_m * l_n)
					from b_idx := 0 until b_idx >= l_batch_count loop
						from i := 0 until i >= l_m loop
							from j := 0 until j >= l_n loop
								l_dot_int := 0
								from p := 0 until p >= l_k loop
									l_a_int := storage_item_as_int_64_universal (storage, offset + b_idx * l_m * l_k + i * l_k + p + 1)
									l_b_int := storage_item_as_int_64_universal (other.storage, other.offset + b_idx * l_k * l_n + p * l_n + j + 1)
									l_dot_int := l_dot_int + l_a_int * l_b_int
									p := p + 1
								end
								l_res_store_int.put_int_64 (l_dot_int, b_idx * l_m * l_n + i * l_n + j + 1)
								j := j + 1
							end
							i := i + 1
						end
						b_idx := b_idx + 1
					end
					create Result.make_from_storage (l_res_store_int, l_res_shape, l_res_strides, 0)
				else
					create l_res_store_real.make (l_batch_count * l_m * l_n)
					from b_idx := 0 until b_idx >= l_batch_count loop
						from i := 0 until i >= l_m loop
							from j := 0 until j >= l_n loop
								l_dot_real := 0.0
								from p := 0 until p >= l_k loop
									l_a_val := storage_item_as_real_64_universal (storage, offset + b_idx * l_m * l_k + i * l_k + p + 1)
									l_b_val := storage_item_as_real_64_universal (other.storage, other.offset + b_idx * l_k * l_n + p * l_n + j + 1)
									l_dot_real := l_dot_real + l_a_val * l_b_val
									p := p + 1
								end
								l_res_store_real.put_real_64 (l_dot_real, b_idx * l_m * l_n + i * l_n + j + 1)
								j := j + 1
							end
							i := i + 1
						end
						b_idx := b_idx + 1
					end
					create Result.make_from_storage (l_res_store_real, l_res_shape, l_res_strides, 0)
				end
			end
		end

	storage_item_as_real_64_universal (a_storage: ET_STORAGE; index: INTEGER_32): REAL_64
			-- Read any storage element as REAL_64, regardless of the underlying type.
			-- This is the type-bridging helper that makes the generic fallback work.
		do
			if attached {ET_STORAGE_REAL_64} a_storage as s then
				Result := s.item_as_real_64 (index)
			elseif attached {ET_STORAGE_REAL_32} a_storage as s then
				Result := s.item_as_real_32 (index).to_double
			elseif attached {ET_STORAGE_INT_64} a_storage as s then
				Result := s.item_as_int_64 (index).to_double
			elseif attached {ET_STORAGE_INT_32} a_storage as s then
				Result := s.item_as_int_32 (index).to_double
			elseif attached {ET_STORAGE_BOOL} a_storage as s then
				if s.item_as_boolean (index) then Result := 1.0 else Result := 0.0 end
			end
		end

	storage_item_as_int_64_universal (a_storage: ET_STORAGE; index: INTEGER_32): INTEGER_64
			-- Read any storage element as INTEGER_64, safely, for generic integer matmul.
		do
			if attached {ET_STORAGE_INT_64} a_storage as s then
				Result := s.item_as_int_64 (index)
			elseif attached {ET_STORAGE_INT_32} a_storage as s then
				Result := s.item_as_int_32 (index).to_integer_64
			elseif attached {ET_STORAGE_REAL_64} a_storage as s then
				Result := s.item_as_real_64 (index).truncated_to_integer_64
			elseif attached {ET_STORAGE_REAL_32} a_storage as s then
				Result := s.item_as_real_32 (index).truncated_to_integer_64
			elseif attached {ET_STORAGE_BOOL} a_storage as s then
				if s.item_as_boolean (index) then Result := 1 else Result := 0 end
			end
		end

feature -- Math Operations (continued)

	mul alias "*" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise multiplication.
		require
			same_shape: is_broadcastable (other.shape)

		local
			v1, v2: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_MUL_FUNCTION
		do
			if {ET_TORCH}.is_grad_enabled and then (requires_grad or other.requires_grad) then
				v1 := ensure_grad_node
				v2 := other.ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1, v2>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := mul_internal (other)
			end
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	mul_internal (other: ET_TENSOR): ET_TENSOR
			-- Numeric implementation of multiplication.
		local
			l_strides, br_shape: ARRAY [INTEGER_32]
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_count, i, idx_self, idx_other: INTEGER_32
			l_promoter: ET_DTYPE_PROMOTER
			l_target_dtype: ET_DTYPE
			l_self, l_other: ET_TENSOR
		do
			create l_promoter
			l_target_dtype := l_promoter.promoted_dtype (dtype, other.dtype)
			l_self := if dtype.is_equal (l_target_dtype) then Current else to_dtype (l_target_dtype) end
			l_other := if other.dtype.is_equal (l_target_dtype) then other else other.to_dtype (l_target_dtype) end

			br_shape := broadcast_shape (l_self.shape, l_other.shape)
			l_count := calculate_product (br_shape)
			l_strides := calculate_contiguous_strides (br_shape)

			if l_target_dtype.is_float64 then
				create l_store_f64.make (l_count)
				l_store := l_store_f64
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_f64.put_real_64 (l_self.storage.item_as_real_64 (l_self.offset + idx_self + 1) * l_other.storage.item_as_real_64 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			elseif l_target_dtype.is_float32 then
				create l_store_f32.make (l_count)
				l_store := l_store_f32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_f32.put_real_32 (l_self.storage.item_as_real_32 (l_self.offset + idx_self + 1) * l_other.storage.item_as_real_32 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			elseif l_target_dtype.is_int32 then
				create {ET_STORAGE_INT_32} l_store.make (l_count)
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					if attached {ET_STORAGE_INT_32} l_store as s then
						s.put_int_32 (l_self.storage.item_as_int_32 (l_self.offset + idx_self + 1) * l_other.storage.item_as_int_32 (l_other.offset + idx_other + 1), i)
					end
					i := i + 1
				end
			elseif l_target_dtype.is_int64 then
				if attached {ET_STORAGE_INT_64} l_self.storage as target_store and then
				   attached {ET_STORAGE_INT_64} l_other.storage as src_store then
					create {ET_STORAGE_INT_64} l_store.make (l_count)
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
						idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
						if attached {ET_STORAGE_INT_64} l_store as s then
							s.put_int_64 (target_store.item_as_int_64 (l_self.offset + idx_self + 1) * src_store.item_as_int_64 (l_other.offset + idx_other + 1), i)
						end
						i := i + 1
					end
				else
					(create {EXCEPTIONS}).raise ("Storage mismatch for int64 multiplication")
					create {ET_STORAGE_REAL_64} l_store.make (0)
				end
			elseif l_target_dtype.is_bool then
				if attached {ET_STORAGE_BOOL} l_self.storage as target_store and then
				   attached {ET_STORAGE_BOOL} l_other.storage as src_store then
					create {ET_STORAGE_BOOL} l_store.make (l_count)
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
						idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
						if attached {ET_STORAGE_BOOL} l_store as s then
							s.put_boolean (target_store.item_as_boolean (l_self.offset + idx_self + 1) and src_store.item_as_boolean (l_other.offset + idx_other + 1), i)
						end
						i := i + 1
					end
				else
					(create {EXCEPTIONS}).raise ("Storage mismatch for bool multiplication")
					create {ET_STORAGE_REAL_64} l_store.make (0)
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for multiplication: " + l_target_dtype.out)
				create {ET_STORAGE_REAL_64} l_store.make (0)
			end

			create Result.make_from_storage (l_store, br_shape, l_strides, 0)
			Result.set_dtype (l_target_dtype)
		end

	mul_in_place (other: ET_TENSOR)
			-- In-place element-wise multiplication.
		require
			not_requires_grad: not requires_grad
			same_shape: is_broadcastable (other.shape)

		local
			br_shape: ARRAY [INTEGER_32]
			l_count, i, idx_self, idx_other: INTEGER_32
		do
			br_shape := broadcast_shape (shape, other.shape)
			l_count := calculate_product (br_shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_real_64 (s.item_as_real_64 (offset + idx_self + 1) * other.storage.item_as_real_64 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_real_32 (s.item_as_real_32 (offset + idx_self + 1) * other.storage.item_as_real_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_int32 then
				if attached {ET_STORAGE_INT_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_int_32 (s.item_as_int_32 (offset + idx_self + 1) * other.storage.item_as_int_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_bool then
				if attached {ET_STORAGE_BOOL} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_boolean (s.item_as_boolean (offset + idx_self + 1) and other.storage.item_as_boolean (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for mul_in_place")
			end
		end

	mul_scalar (val: REAL_64): ET_TENSOR
			-- Element-wise multiplication with a scalar.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			l_strides := calculate_contiguous_strides (shape)

			if dtype.is_float64 then
				create l_store_f64.make (l_count)
				l_store := l_store_f64
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, shape, strides, shape)
					l_store_f64.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) * val, i)
					i := i + 1
				end
			elseif dtype.is_float32 then
				create l_store_f32.make (l_count)
				l_store := l_store_f32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, shape, strides, shape)
					l_store_f32.put_real_32 (storage.item_as_real_32 (offset + idx_self + 1) * val.truncated_to_real, i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("mul_scalar only supports float tensors currently")
				create l_store_f64.make (0)
				l_store := l_store_f64
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
			Result.set_dtype (dtype)
		end

	mul_scalar_in_place (val: REAL_64)
			-- In-place element-wise multiplication with a scalar.
		require
			not_requires_grad: not requires_grad
		local
			l_count, i, idx_self: INTEGER_32

		do
			l_count := calculate_product (shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						s.put_real_64 (s.item_as_real_64 (offset + idx_self + 1) * val, offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						s.put_real_32 (s.item_as_real_32 (offset + idx_self + 1) * val.truncated_to_real, offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				(create {EXCEPTIONS}).raise ("mul_scalar_in_place only supports float tensors currently")
			end
		end

	div alias "/" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise division.
		require
			same_shape: is_broadcastable (other.shape)
		do
			Result := div_internal (other)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	div_internal (other: ET_TENSOR): ET_TENSOR
			-- Numeric implementation of division.
		local
			l_strides, br_shape: ARRAY [INTEGER_32]
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_store_i32: ET_STORAGE_INT_32
			l_count, i, idx_self, idx_other: INTEGER_32
			l_promoter: ET_DTYPE_PROMOTER
			l_target_dtype: ET_DTYPE
			l_self, l_other: ET_TENSOR
		do
			create l_promoter
			l_target_dtype := l_promoter.promoted_dtype (dtype, other.dtype)
			l_self := if dtype.is_equal (l_target_dtype) then Current else to_dtype (l_target_dtype) end
			l_other := if other.dtype.is_equal (l_target_dtype) then other else other.to_dtype (l_target_dtype) end

			br_shape := broadcast_shape (l_self.shape, l_other.shape)
			l_count := calculate_product (br_shape)
			l_strides := calculate_contiguous_strides (br_shape)

			if l_target_dtype.is_float64 then
				create l_store_f64.make (l_count)
				l_store := l_store_f64
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_f64.put_real_64 (l_self.storage.item_as_real_64 (l_self.offset + idx_self + 1) / l_other.storage.item_as_real_64 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			elseif l_target_dtype.is_float32 then
				create l_store_f32.make (l_count)
				l_store := l_store_f32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_f32.put_real_32 (l_self.storage.item_as_real_32 (l_self.offset + idx_self + 1) / l_other.storage.item_as_real_32 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			elseif l_target_dtype.is_int32 then
				create l_store_i32.make (l_count)
				l_store := l_store_i32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, l_self.shape, l_self.strides, br_shape)
					idx_other := linear_index_to_offset (i, l_other.shape, l_other.strides, br_shape)
					l_store_i32.put_int_32 (l_self.storage.item_as_int_32 (l_self.offset + idx_self + 1) // l_other.storage.item_as_int_32 (l_other.offset + idx_other + 1), i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for division")
				create l_store_f64.make (0)
				l_store := l_store_f64
			end

			create Result.make_from_storage (l_store, br_shape, l_strides, 0)
			Result.set_dtype (l_target_dtype)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	div_in_place (other: ET_TENSOR)
			-- In-place element-wise division.
		require
			not_requires_grad: not requires_grad
			same_shape: is_broadcastable (other.shape)

		local
			br_shape: ARRAY [INTEGER_32]
			l_count, i, idx_self, idx_other: INTEGER_32

		do
			br_shape := broadcast_shape (shape, other.shape)
			l_count := calculate_product (br_shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_real_64 (s.item_as_real_64 (offset + idx_self + 1) / other.storage.item_as_real_64 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_real_32 (s.item_as_real_32 (offset + idx_self + 1) / other.storage.item_as_real_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_int32 then
				if attached {ET_STORAGE_INT_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, br_shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
						s.put_int_32 (s.item_as_int_32 (offset + idx_self + 1) // other.storage.item_as_int_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for div_in_place")
			end
		end

	div_scalar (val: REAL_64): ET_TENSOR
			-- Element-wise division by a scalar.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			l_strides := calculate_contiguous_strides (shape)

			if dtype.is_float64 then
				create l_store_f64.make (l_count)
				l_store := l_store_f64
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, shape, strides, shape)
					l_store_f64.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) / val, i)
					i := i + 1
				end
			elseif dtype.is_float32 then
				create l_store_f32.make (l_count)
				l_store := l_store_f32
				from i := 1 until i > l_count loop
					idx_self := linear_index_to_offset (i, shape, strides, shape)
					l_store_f32.put_real_32 (storage.item_as_real_32 (offset + idx_self + 1) / val.truncated_to_real, i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("div_scalar only supports float tensors currently")
				create l_store_f64.make (0)
				l_store := l_store_f64
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
			Result.set_dtype (dtype)
		end

	div_scalar_in_place (val: REAL_64)
			-- In-place element-wise division by a scalar.
		require
			not_requires_grad: not requires_grad
		local
			l_count, i, idx_self: INTEGER_32

		do
			l_count := calculate_product (shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						s.put_real_64 (s.item_as_real_64 (offset + idx_self + 1) / val, offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as s then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						s.put_real_32 (s.item_as_real_32 (offset + idx_self + 1) / val.truncated_to_real, offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				(create {EXCEPTIONS}).raise ("div_scalar_in_place only supports float tensors currently")
			end
		end

	exp_val: ET_TENSOR
			-- Element-wise exponential.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_store.put_real_64 (l_math.exp (storage.item_as_real_64 (offset + idx_self + 1)), i)
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	sqrt_val: ET_TENSOR
			-- Element-wise square root.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_store.put_real_64 (l_math.sqrt (storage.item_as_real_64 (offset + idx_self + 1)), i)
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	log_val: ET_TENSOR
			-- Element-wise natural logarithm.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_store.put_real_64 (l_math.log (storage.item_as_real_64 (offset + idx_self + 1)), i)
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	tanh_val: ET_TENSOR
			-- Element-wise hyperbolic tangent.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_math: DOUBLE_MATH
			l_val, l_t: REAL_64
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_val := storage.item_as_real_64 (offset + idx_self + 1)
				l_t := tanh_scalar (l_val)
				l_store.put_real_64 (l_t, i)
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	gelu: ET_TENSOR
			-- Gaussian Error Linear Unit (element-wise).
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_GELU_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := gelu_internal
			end
		end

	gelu_internal: ET_TENSOR
			-- Numeric implementation of GELU.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_val, l_cdf: REAL_64
			pi, l_sqrt_2_over_pi: REAL_64
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math
			pi := 3.14159265358979323846
			l_sqrt_2_over_pi := l_math.sqrt (2.0 / pi)

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_val := storage.item_as_real_64 (offset + idx_self + 1)
				l_cdf := 0.5 * (1.0 + tanh_scalar (l_sqrt_2_over_pi * (l_val + 0.044715 * l_val * l_val * l_val)))
				l_store.put_real_64 (l_val * l_cdf, i)
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape.deep_twin, l_strides, 0)
			Result.set_dtype (dtype)
		end

	relu: ET_TENSOR
			-- Rectified Linear Unit (element-wise), autograd-aware.
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_RELU_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := relu_internal
			end
		end

	relu_internal: ET_TENSOR
			-- Numeric implementation of ReLU: max(0, x).
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_val: REAL_64
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_val := storage.item_as_real_64 (offset + idx_self + 1)
				if l_val > 0.0 then
					l_store.put_real_64 (l_val, i)
				else
					l_store.put_real_64 (0.0, i)
				end
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape.deep_twin, l_strides, 0)
			Result.set_dtype (dtype)
		end

	relu_mask_internal: ET_TENSOR
			-- Binary mask: 1.0 where x > 0, else 0.0. Used by ET_RELU_FUNCTION.backward.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_val: REAL_64
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_val := storage.item_as_real_64 (offset + idx_self + 1)
				if l_val > 0.0 then
					l_store.put_real_64 (1.0, i)
				else
					l_store.put_real_64 (0.0, i)
				end
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape.deep_twin, l_strides, 0)
			Result.set_dtype (dtype)
		end

	sigmoid: ET_TENSOR
			-- Sigmoid activation σ(x) = 1 / (1 + exp(-x)), autograd-aware.
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_SIGMOID_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := sigmoid_internal
			end
		end

	sigmoid_internal: ET_TENSOR
			-- Numeric implementation of Sigmoid: 1 / (1 + exp(-x)).
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
			l_val: REAL_64
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math

			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_val := storage.item_as_real_64 (offset + idx_self + 1)
				l_store.put_real_64 (1.0 / (1.0 + l_math.exp (-l_val)), i)
				i := i + 1
			end

			create Result.make_from_storage (l_store, shape.deep_twin, l_strides, 0)
			Result.set_dtype (dtype)
		end

	tanh: ET_TENSOR
			-- Hyperbolic tangent (element-wise), autograd-aware.
			-- Use tanh_val for the raw numeric version.
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_TANH_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := tanh_val
			end
		end

feature -- Reductions

	sum_to_size (target_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Reduce tensor to match `target_shape` by summing across broadcasted dimensions.
			-- Useful for autograd backward passes involving broadcasting.
		require
			target_shape_valid: target_shape.count <= rank
		local
			l_res: ET_TENSOR
			diff: INTEGER_32
			i: INTEGER_32
			d: INTEGER_32
		do
			l_res := Current
			diff := rank - target_shape.count

			-- 1. Sum away prepended dimensions (e.g., shape (2, 3, 4) -> target (3, 4), sum dim 1)
			from i := 1 until i > diff loop
				-- Always sum the first dimension since the rank decreases each time
				l_res := l_res.sum_dim (1, False)
				i := i + 1
			end

			-- 2. Sum away broadcasted dimensions (e.g., shape (3, 4) -> target (1, 4), sum dim 1 keep_dim=True)
			from d := 1 until d > target_shape.count loop
				if l_res.shape [d] > 1 and target_shape [d] = 1 then
					l_res := l_res.sum_dim (d, True)
				end
				d := d + 1
			end

			Result := l_res
		end

	sum (dims: ARRAY [INTEGER_32]; keep_dim: BOOLEAN): ET_TENSOR
			-- Sum reduction over specific dimensions.
		require
			valid_dims: not dims.is_empty
		local
			i, j, temp: INTEGER_32
			l_res: ET_TENSOR
			sorted_dims: ARRAY [INTEGER_32]
		do
			sorted_dims := dims.deep_twin
			from i := 1 until i > sorted_dims.count loop
				from j := 1 until j > sorted_dims.count - i loop
					if sorted_dims [j] < sorted_dims [j + 1] then
						temp := sorted_dims [j]
						sorted_dims [j] := sorted_dims [j + 1]
						sorted_dims [j + 1] := temp
					end
					j := j + 1
				end
				i := i + 1
			end

			l_res := Current
			from i := 1 until i > sorted_dims.count loop
				l_res := l_res.sum_dim (sorted_dims [i], keep_dim)
				i := i + 1
			end
			Result := l_res
		end

	mean (dims: ARRAY [INTEGER_32]; keep_dim: BOOLEAN): ET_TENSOR
			-- Mean reduction over specific dimensions.
		require
			valid_dims: not dims.is_empty
		local
			i, j, temp: INTEGER_32
			l_res: ET_TENSOR
			sorted_dims: ARRAY [INTEGER_32]
		do
			sorted_dims := dims.deep_twin
			from i := 1 until i > sorted_dims.count loop
				from j := 1 until j > sorted_dims.count - i loop
					if sorted_dims [j] < sorted_dims [j + 1] then
						temp := sorted_dims [j]
						sorted_dims [j] := sorted_dims [j + 1]
						sorted_dims [j + 1] := temp
					end
					j := j + 1
				end
				i := i + 1
			end

			l_res := Current
			from i := 1 until i > sorted_dims.count loop
				l_res := l_res.mean_dim (sorted_dims [i], keep_dim)
				i := i + 1
			end
			Result := l_res
		end

	max (dims: ARRAY [INTEGER_32]; keep_dim: BOOLEAN): ET_TENSOR
			-- Max reduction over specific dimensions.
		require
			valid_dims: not dims.is_empty
		local
			i, j, temp: INTEGER_32
			l_res: ET_TENSOR
			sorted_dims: ARRAY [INTEGER_32]
		do
			sorted_dims := dims.deep_twin
			from i := 1 until i > sorted_dims.count loop
				from j := 1 until j > sorted_dims.count - i loop
					if sorted_dims [j] < sorted_dims [j + 1] then
						temp := sorted_dims [j]
						sorted_dims [j] := sorted_dims [j + 1]
						sorted_dims [j + 1] := temp
					end
					j := j + 1
				end
				i := i + 1
			end

			l_res := Current
			from i := 1 until i > sorted_dims.count loop
				l_res := l_res.max_dim (sorted_dims [i], keep_dim)
				i := i + 1
			end
			Result := l_res
		end

	sum_dim (dim: INTEGER_32; keep_dim: BOOLEAN): ET_TENSOR
			-- Sum reduction over a specific dimension.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_SUM_DIM_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward_with_params (<<v1>>, dim, keep_dim)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := sum_dim_internal (dim, keep_dim)
			end
		end

	sum_dim_internal (dim: INTEGER_32; keep_dim: BOOLEAN): ET_TENSOR
			-- Numeric implementation of sum_dim.
		local
			l_res_shape, l_final_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE
			l_res_store_f64: ET_STORAGE_REAL_64
			l_res_store_f32: ET_STORAGE_REAL_32
			l_res_store_i32: ET_STORAGE_INT_32
			i, k, idx_src: INTEGER_32
			l_sum_f64: REAL_64
			l_sum_f32: REAL_32
			l_sum_i32: INTEGER_32
			l_count: INTEGER_32
		do
			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1
			l_count := calculate_product (l_res_shape)

			if dtype.is_float64 then
				create l_res_store_f64.make (l_count)
				l_res_store := l_res_store_f64
				from i := 1 until i > l_count loop
					idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_sum_f64 := 0.0
					from k := 0 until k >= shape [dim] loop
						l_sum_f64 := l_sum_f64 + storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
						k := k + 1
					end
					l_res_store_f64.put_real_64 (l_sum_f64, i)
					i := i + 1
				end
			elseif dtype.is_float32 then
				create l_res_store_f32.make (l_count)
				l_res_store := l_res_store_f32
				from i := 1 until i > l_count loop
					idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_sum_f32 := 0.0
					from k := 0 until k >= shape [dim] loop
						l_sum_f32 := l_sum_f32 + storage.item_as_real_32 (offset + idx_src + k * strides [dim] + 1)
						k := k + 1
					end
					l_res_store_f32.put_real_32 (l_sum_f32, i)
					i := i + 1
				end
			elseif dtype.is_int32 then
				create l_res_store_i32.make (l_count)
				l_res_store := l_res_store_i32
				from i := 1 until i > l_count loop
					idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_sum_i32 := 0
					from k := 0 until k >= shape [dim] loop
						l_sum_i32 := l_sum_i32 + storage.item_as_int_32 (offset + idx_src + k * strides [dim] + 1)
						k := k + 1
					end
					l_res_store_i32.put_int_32 (l_sum_i32, i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for sum_dim")
				create l_res_store_f64.make (0)
				l_res_store := l_res_store_f64
			end

			if keep_dim then
				l_final_shape := l_res_shape
			else
				create l_final_shape.make_empty
				from k := 1 until k > l_res_shape.count loop
					if k /= dim then
						l_final_shape.force (l_res_shape [k], l_final_shape.count + 1)
					end
					k := k + 1
				end
				if l_final_shape.is_empty then
					l_final_shape.force (1, 1)
				end
			end

			l_res_strides := calculate_contiguous_strides (l_final_shape)
			create Result.make_from_storage (l_res_store, l_final_shape, l_res_strides, 0)
			Result.set_dtype (dtype)
		ensure
			valid_result: Result /= Void
		end

	mean_dim (dim: INTEGER_32; keep_dim: BOOLEAN): ET_TENSOR
			-- Mean reduction over a specific dimension.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_res: ET_TENSOR
			l_N: REAL_64
		do
			l_res := sum_dim (dim, keep_dim)
			l_N := shape [dim].to_double
			Result := l_res.mul_scalar (1.0 / l_N)
		end

	var_dim (dim: INTEGER_32; unbiased: BOOLEAN; keep_dim: BOOLEAN): ET_TENSOR
			-- Variance reduction over a specific dimension.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_mean: ET_TENSOR
			l_res_shape, l_final_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_inner, i, k, idx_src, idx_res: INTEGER_32
			l_store: ET_STORAGE_REAL_64
			mean_val, x_val, sum_sq: REAL_64
			n_div: REAL_64
			l_count: INTEGER_32
		do
			l_mean := mean_dim (dim, True) -- Keepdim for broadcasting implicitly

			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1
			l_count := calculate_product (l_res_shape)

			l_inner := shape [dim]

			if unbiased then
				n_div := (l_inner - 1).to_double
			else
				n_div := l_inner.to_double
			end
			if n_div < 1.0 then n_div := 1.0 end

			create l_store.make (l_count)
			from i := 1 until i > l_count loop
				idx_res := linear_index_to_offset (i, l_res_shape, calculate_contiguous_strides(l_res_shape), l_res_shape)
				idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)

				-- The mean tensor has shape `l_res_shape`. Its strides are contiguous.
				mean_val := storage_item_as_real_64_universal (l_mean.storage, l_mean.offset + idx_res + 1)
				sum_sq := 0.0
				from k := 0 until k >= l_inner loop
					x_val := storage_item_as_real_64_universal (storage, offset + idx_src + k * strides [dim] + 1)
					sum_sq := sum_sq + (x_val - mean_val) * (x_val - mean_val)
					k := k + 1
				end
				l_store.put_real_64 (sum_sq / n_div, i)
				i := i + 1
			end

			if keep_dim then
				l_final_shape := l_res_shape
			else
				create l_final_shape.make_empty
				from k := 1 until k > l_res_shape.count loop
					if k /= dim then
						l_final_shape.force (l_res_shape [k], l_final_shape.count + 1)
					end
					k := k + 1
				end
				if l_final_shape.is_empty then l_final_shape.force (1, 1) end
			end

			l_res_strides := calculate_contiguous_strides (l_final_shape)
			create Result.make_from_storage (l_store, l_final_shape, l_res_strides, 0)
			Result.set_dtype (dtype)
		end

	max_dim (dim: INTEGER_32; keep_dim: BOOLEAN): ET_TENSOR
			-- Max reduction over a specific dimension.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_res_shape, l_final_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE
			l_res_store_f64: ET_STORAGE_REAL_64
			l_res_store_f32: ET_STORAGE_REAL_32
			l_res_store_i32: ET_STORAGE_INT_32
			i, k, idx_src: INTEGER_32
			l_max_f64, l_val_f64: REAL_64
			l_max_f32, l_val_f32: REAL_32
			l_max_i32, l_val_i32: INTEGER_32
			l_count: INTEGER_32
		do
			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1

			l_count := calculate_product (l_res_shape)

			if dtype.is_float64 then
				create l_res_store_f64.make (l_count)
				l_res_store := l_res_store_f64
				from i := 1 until i > l_count loop
					idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_max_f64 := -1.7976931348623157e+308
					from k := 0 until k >= shape [dim] loop
						l_val_f64 := storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
						if l_val_f64 > l_max_f64 or k = 0 then l_max_f64 := l_val_f64 end
						k := k + 1
					end
					l_res_store_f64.put_real_64 (l_max_f64, i)
					i := i + 1
				end
			elseif dtype.is_float32 then
				create l_res_store_f32.make (l_count)
				l_res_store := l_res_store_f32
				from i := 1 until i > l_count loop
					idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_max_f32 := {REAL_32} -3.4028235e+38
					from k := 0 until k >= shape [dim] loop
						l_val_f32 := storage.item_as_real_32 (offset + idx_src + k * strides [dim] + 1)
						if l_val_f32 > l_max_f32 or k = 0 then l_max_f32 := l_val_f32 end
						k := k + 1
					end
					l_res_store_f32.put_real_32 (l_max_f32, i)
					i := i + 1
				end
			elseif dtype.is_int32 then
				create l_res_store_i32.make (l_count)
				l_res_store := l_res_store_i32
				from i := 1 until i > l_count loop
					idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
					l_max_i32 := {INTEGER_32}.min_value
					from k := 0 until k >= shape [dim] loop
						l_val_i32 := storage.item_as_int_32 (offset + idx_src + k * strides [dim] + 1)
						if l_val_i32 > l_max_i32 or k = 0 then l_max_i32 := l_val_i32 end
						k := k + 1
					end
					l_res_store_i32.put_int_32 (l_max_i32, i)
					i := i + 1
				end
			else
				(create {EXCEPTIONS}).raise ("Unsupported dtype for max_dim")
				create l_res_store_f64.make (0)
				l_res_store := l_res_store_f64
			end

			if keep_dim then
				l_final_shape := l_res_shape
			else
				create l_final_shape.make_empty
				from k := 1 until k > l_res_shape.count loop
					if k /= dim then
						l_final_shape.force (l_res_shape [k], l_final_shape.count + 1)
					end
					k := k + 1
				end
				if l_final_shape.is_empty then l_final_shape.force (1, 1) end
			end

			l_res_strides := calculate_contiguous_strides (l_final_shape)
			create Result.make_from_storage (l_res_store, l_final_shape, l_res_strides, 0)
			Result.set_dtype (dtype)
		end

	softmax (dim: INTEGER_32): ET_TENSOR
			-- Softmax over dimension `dim`.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_SOFTMAX_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward_with_dim (<<v1>>, dim)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := softmax_internal (dim)
			end
		end

	softmax_internal (dim: INTEGER_32): ET_TENSOR
			-- Numeric implementation of softmax.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_max, l_sum, l_val: REAL_64
			i, k, idx_src, idx_dst: INTEGER_32
			l_res_store: ET_STORAGE_REAL_64
			l_res_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_math: DOUBLE_MATH
			l_count: INTEGER_32
		do
			create l_math
			l_count := calculate_product (shape)
			create l_res_store.make (l_count)
			l_res_strides := calculate_contiguous_strides (shape)

			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1

			from i := 1 until i > calculate_product (l_res_shape) loop
				idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
				idx_dst := linear_index_to_offset (i, shape, l_res_strides, l_res_shape)

				l_max := storage.item_as_real_64 (offset + idx_src + 1)
				from k := 1 until k >= shape [dim] loop
					l_val := storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
					if l_val > l_max then l_max := l_val end
					k := k + 1
				end

				l_sum := 0.0
				from k := 0 until k >= shape [dim] loop
					l_val := storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
					l_val := l_math.exp (l_val - l_max)
					l_sum := l_sum + l_val
					l_res_store.put_real_64 (l_val, idx_dst + k * l_res_strides [dim] + 1)
					k := k + 1
				end

				from k := 0 until k >= shape [dim] loop
					l_val := l_res_store.item_as_real_64 (idx_dst + k * l_res_strides [dim] + 1)
					l_res_store.put_real_64 (l_val / l_sum, idx_dst + k * l_res_strides [dim] + 1)
					k := k + 1
				end

				i := i + 1
			end

			create Result.make_from_storage (l_res_store, shape.deep_twin, l_res_strides, 0)
			Result.set_dtype (dtype)
		end

	rms_norm (dim: INTEGER_32; eps: REAL_64): ET_TENSOR
			-- Root Mean Square Normalization over dimension `dim`.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_sum, l_val, l_rms: REAL_64
			i, k, idx_src, idx_dst: INTEGER_32
			l_res_store: ET_STORAGE_REAL_64
			l_res_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_math: DOUBLE_MATH
			l_count: INTEGER_32
			l_n: REAL_64
		do
			create l_math
			l_count := calculate_product (shape)
			create l_res_store.make (l_count)
			l_res_strides := calculate_contiguous_strides (shape)

			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1
			l_n := shape [dim]

			from i := 1 until i > calculate_product (l_res_shape) loop
				idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
				idx_dst := linear_index_to_offset (i, shape, l_res_strides, l_res_shape)

				l_sum := 0.0
				from k := 0 until k >= shape [dim] loop
					l_val := storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
					l_sum := l_sum + (l_val * l_val)
					k := k + 1
				end

				l_rms := l_math.sqrt (l_sum / l_n + eps)

				from k := 0 until k >= shape [dim] loop
					l_val := storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
					l_res_store.put_real_64 (l_val / l_rms, idx_dst + k * l_res_strides [dim] + 1)
					k := k + 1
				end

				i := i + 1
			end

			create Result.make_from_storage (l_res_store, shape.deep_twin, l_res_strides, 0)
		end

	mean_all: ET_TENSOR
			-- Global mean of all elements in the tensor. Returns a scalar tensor.
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_MEAN_ALL_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward (<<v1>>)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := mean_all_internal
			end
		end

	mean_all_internal: ET_TENSOR
			-- Numeric implementation of mean_all.
			-- Correctly handles non-contiguous tensors (after transpose/slice) via strides.
		local
			l_sum: REAL_64
			i, idx: INTEGER_32
			l_count: INTEGER_32
			l_store: ET_STORAGE_REAL_64
		do
			l_sum := 0.0
			l_count := calculate_product (shape)
			from i := 1 until i > l_count loop
				idx := linear_index_to_offset (i, shape, strides, shape)
				l_sum := l_sum + storage_item_as_real_64_universal (storage, offset + idx + 1)
				i := i + 1
			end

			create l_store.make (1)
			l_store.put_real_64 (l_sum / l_count, 1)
			create Result.make_from_storage (l_store, <<1>>, <<1>>, 0)
		end


feature -- Views (Zero-copy or copy-on-non-contiguous)

	is_contiguous: BOOLEAN
			-- Is this tensor stored in contiguous C-order (row-major) memory?
		local
			l_strides: ARRAY [INTEGER_32]
		do
			l_strides := calculate_contiguous_strides (shape)
			Result := strides ~ l_strides
		end

	contiguous: ET_TENSOR
			-- Return a contiguous copy of this tensor if non-contiguous, else return Current.
		local
			l_store: ET_STORAGE
			l_store_f64: ET_STORAGE_REAL_64
			l_store_f32: ET_STORAGE_REAL_32
			l_store_i32: ET_STORAGE_INT_32
			l_count, i, idx: INTEGER_32
			l_strides: ARRAY [INTEGER_32]
		do
			if is_contiguous then
				Result := Current
			else
				l_count := calculate_product (shape)
				l_strides := calculate_contiguous_strides (shape)
				if dtype.is_float64 then
					create l_store_f64.make (l_count)
					l_store := l_store_f64
					from i := 1 until i > l_count loop
						idx := linear_index_to_offset (i, shape, strides, shape)
						l_store_f64.put_real_64 (storage.item_as_real_64 (offset + idx + 1), i)
						i := i + 1
					end
				elseif dtype.is_float32 then
					create l_store_f32.make (l_count)
					l_store := l_store_f32
					from i := 1 until i > l_count loop
						idx := linear_index_to_offset (i, shape, strides, shape)
						l_store_f32.put_real_32 (storage.item_as_real_32 (offset + idx + 1), i)
						i := i + 1
					end
				elseif dtype.is_int32 then
					create l_store_i32.make (l_count)
					l_store := l_store_i32
					from i := 1 until i > l_count loop
						idx := linear_index_to_offset (i, shape, strides, shape)
						l_store_i32.put_int_32 (storage.item_as_int_32 (offset + idx + 1), i)
						i := i + 1
					end
				else
					-- bool and int64 fallback via universal reader
					create l_store_f64.make (l_count)
					l_store := l_store_f64
					from i := 1 until i > l_count loop
						idx := linear_index_to_offset (i, shape, strides, shape)
						l_store_f64.put_real_64 (storage_item_as_real_64_universal (storage, offset + idx + 1), i)
						i := i + 1
					end
				end
				create Result.make_from_storage (l_store, shape.deep_twin, l_strides, 0)
				Result.set_dtype (dtype)
			end
		ensure
			result_contiguous: Result.is_contiguous
		end

	view (new_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Return a new tensor with the same data but different shape.
			-- Requires the tensor to be contiguous in memory; raises a precondition
			-- violation otherwise. Use `reshape` to handle non-contiguous tensors safely.
		require
			view_legal: calculate_product (shape) = calculate_product (new_shape)
			is_contiguous: is_contiguous
		local
			l_strides: ARRAY [INTEGER_32]
		do
			l_strides := calculate_contiguous_strides (new_shape)
			create Result.make_from_storage (storage, new_shape, l_strides, offset)
			Result.set_dtype (dtype)
		ensure
			zero_copy: Result.storage = storage
			shape_set: Result.shape ~ new_shape
		end

	reshape (a_new_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Return a tensor with the new shape.
			-- If the tensor is contiguous, returns a zero-copy view.
			-- If non-contiguous (e.g. after transpose), makes a compact copy first.
		require
			valid_reshape: calculate_product (shape) = calculate_product (a_new_shape)
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_RESHAPE_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward_with_params (<<v1>>, a_new_shape)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := reshape_internal (a_new_shape)
			end
		ensure
			shape_set: Result.shape ~ a_new_shape
		end

	reshape_internal (a_new_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Numeric implementation of reshape.
		do
			Result := contiguous.view (a_new_shape)
		end

	transpose (dim1, dim2: INTEGER_32): ET_TENSOR
			-- Returns a tensor that is a transposed version of `Current`.
			-- (0-based mathematically, adapt internally to 1-based Eiffel structs).
		require
			valid_dim1: dim1 >= 1 and dim1 <= rank
			valid_dim2: dim2 >= 1 and dim2 <= rank
		local
			v1: ET_VALUE
			res_v: ET_VALUE
			l_func: ET_TRANSPOSE_FUNCTION
		do
			if requires_grad then
				v1 := ensure_grad_node
				create l_func
				res_v := l_func.forward_with_params (<<v1>>, dim1, dim2)
				Result := res_v.data
				Result.set_requires_grad (True)
				Result.set_grad_node (res_v)
			else
				Result := transpose_internal (dim1, dim2)
			end
		end

	transpose_internal (dim1, dim2: INTEGER_32): ET_TENSOR
			-- Numeric implementation.
		local
			l_shape: ARRAY [INTEGER_32]
			l_strides: ARRAY [INTEGER_32]
			t_shape, t_stride: INTEGER_32
		do
			l_shape := shape.deep_twin
			l_strides := strides.deep_twin

			t_shape := l_shape [dim1]
			l_shape [dim1] := l_shape [dim2]
			l_shape [dim2] := t_shape

			t_stride := l_strides [dim1]
			l_strides [dim1] := l_strides [dim2]
			l_strides [dim2] := t_stride

			create Result.make_from_storage (storage, l_shape, l_strides, offset)
		ensure
			zero_copy: Result.storage = storage
			shape_swapped: Result.shape [dim1] = shape [dim2] and Result.shape [dim2] = shape [dim1]
		end

	slice_range (dim, start_idx, length: INTEGER_32): ET_TENSOR
			-- Returns a narrow view of the tensor. Equivalent to `narrow`.
		require
			valid_dim: dim >= 1 and dim <= rank
			valid_start: start_idx >= 1
			valid_length: length >= 1
			valid_bounds: start_idx + length - 1 <= shape [dim]
		local
			l_shape: ARRAY [INTEGER_32]
			l_offset: INTEGER_32
		do
			l_shape := shape.deep_twin
			l_shape [dim] := length

			-- Advance offset: start_idx is 1-based
			l_offset := offset + (start_idx - 1) * strides [dim]

			create Result.make_from_storage (storage, l_shape, strides.deep_twin, l_offset)
		ensure
			zero_copy: Result.storage = storage
			shape_updated: Result.shape [dim] = length
		end

feature -- Helpers

	tanh_scalar (l_val: REAL_64): REAL_64
		local
			l_e2x: REAL_64
			l_math: DOUBLE_MATH
		do
			create l_math
			if l_val > 20.0 then
				Result := 1.0
			elseif l_val < -20.0 then
				Result := -1.0
			else
				l_e2x := l_math.exp (2.0 * l_val)
				Result := (l_e2x - 1.0) / (l_e2x + 1.0)
			end
		end

	linear_index_to_offset (linear_idx: INTEGER_32; a_shape: ARRAY [INTEGER_32]; a_strides: ARRAY [INTEGER_32]; br_shape: ARRAY [INTEGER_32]): INTEGER_32
			-- Converts a 1-based linear index from a broadcasted shape into a physical storage offset based on shape and strides.
		local
			rem, dim_idx, i, j: INTEGER_32
		do
			rem := linear_idx - 1
			Result := 0
			j := a_shape.count
			from i := br_shape.count until i < 1 loop
				dim_idx := rem \\ br_shape [i]

				if j >= 1 then
					if a_shape [j] = 1 then
						Result := Result + 0
					else
						Result := Result + dim_idx * a_strides [j]
					end
					j := j - 1
				end

				rem := rem // br_shape [i]
				i := i - 1
			end
		end

	copy_from (other: ET_TENSOR)
			-- Deep copies data from `other` into `Current`.
			-- `Current` and `other` must have perfectly matching shapes.
		require
			same_shape: shape ~ other.shape

		local
			l_count, i, idx_self, idx_other: INTEGER_32
		do
			l_count := calculate_product (shape)

			if dtype.is_float64 then
				if attached {ET_STORAGE_REAL_64} storage as target_store and then
				   attached {ET_STORAGE_REAL_64} other.storage as src_store then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, shape)
						target_store.put_real_64 (src_store.item_as_real_64 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_float32 then
				if attached {ET_STORAGE_REAL_32} storage as target_store and then
				   attached {ET_STORAGE_REAL_32} other.storage as src_store then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, shape)
						target_store.put_real_32 (src_store.item_as_real_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_int32 then
				if attached {ET_STORAGE_INT_32} storage as target_store and then
				   attached {ET_STORAGE_INT_32} other.storage as src_store then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, shape)
						target_store.put_int_32 (src_store.item_as_int_32 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			elseif dtype.is_int64 then
				if attached {ET_STORAGE_INT_64} storage as target_store and then
				   attached {ET_STORAGE_INT_64} other.storage as src_store then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, shape)
						target_store.put_int_64 (src_store.item_as_int_64 (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			else
				if attached {ET_STORAGE_BOOL} storage as target_store and then
				   attached {ET_STORAGE_BOOL} other.storage as src_store then
					from i := 1 until i > l_count loop
						idx_self := linear_index_to_offset (i, shape, strides, shape)
						idx_other := linear_index_to_offset (i, other.shape, other.strides, shape)
						target_store.put_boolean (src_store.item_as_boolean (other.offset + idx_other + 1), offset + idx_self + 1)
						i := i + 1
					end
				end
			end
		end

feature -- Utilities

	calculate_product (arr: ARRAY [INTEGER_32]): INTEGER_32
			-- Return the product of all elements in the array.
		local
			i: INTEGER_32
		do
			Result := 1
			from i := arr.lower until i > arr.upper loop
				Result := Result * arr [i]
				i := i + 1
			end
		end

	calculate_contiguous_strides (a_shape: ARRAY [INTEGER_32]): ARRAY [INTEGER_32]
			-- Calculate contiguous C-style strides.
		local
			i: INTEGER_32
			acc: INTEGER_32
		do
			create Result.make_empty
			if not a_shape.is_empty then
				create Result.make_filled (1, 1, a_shape.count)
				acc := 1
				from i := a_shape.count until i < 1 loop
					Result [i] := acc
					acc := acc * a_shape [i]
					i := i - 1
				end
			end
		end

	is_broadcastable (other_shape: ARRAY [INTEGER_32]): BOOLEAN
			-- Can `Current` and `other` be broadcast together?
		local
			i, j: INTEGER_32
			dim_self, dim_other: INTEGER_32
		do
			Result := True
			i := shape.count
			j := other_shape.count
			from until i < 1 or j < 1 or not Result loop
				dim_self := shape [i]
				dim_other := other_shape [j]
				if dim_self /= dim_other and dim_self /= 1 and dim_other /= 1 then
					Result := False
				end
				i := i - 1
				j := j - 1
			end
		end

	broadcast_shape (shape1, shape2: ARRAY [INTEGER_32]): ARRAY [INTEGER_32]
		local
			i, j, k: INTEGER_32
			dim1, dim2: INTEGER_32
			l_max_rank: INTEGER_32
		do
			l_max_rank := shape1.count.max (shape2.count)
			create Result.make_filled (1, 1, l_max_rank)
			i := shape1.count
			j := shape2.count
			k := l_max_rank
			from until k < 1 loop
				if i >= 1 then dim1 := shape1 [i] else dim1 := 1 end
				if j >= 1 then dim2 := shape2 [j] else dim2 := 1 end
				Result [k] := dim1.max (dim2)
				i := i - 1
				j := j - 1
				k := k - 1
			end
		end


	to_dtype (a_dtype: ET_DTYPE): ET_TENSOR
			-- Convert tensor to new dtype. Returns Current if already of `a_dtype`.
		require
			valid_dtype: a_dtype /= Void
		local
			l_count, i, idx: INTEGER_32
			l_res: ET_TENSOR
			val_real64: REAL_64
		do
			if dtype.is_equal (a_dtype) then
				Result := Current
			else
				create l_res.make_zeros_with_dtype (shape, a_dtype)
				l_count := calculate_product (shape)
				from i := 1 until i > l_count loop
					idx := linear_index_to_offset (i, shape, strides, shape)

					-- Extract as REAL_64
					if dtype.is_float64 then
						if attached {ET_STORAGE_REAL_64} storage as s then val_real64 := s.item_as_real_64 (offset + idx + 1) end
					elseif dtype.is_float32 then
						if attached {ET_STORAGE_REAL_32} storage as s then val_real64 := s.item_as_real_32 (offset + idx + 1) end
					elseif dtype.is_int32 then
						if attached {ET_STORAGE_INT_32} storage as s then val_real64 := s.item_as_int_32 (offset + idx + 1) end
					elseif dtype.is_bool then
						if attached {ET_STORAGE_BOOL} storage as s then
							if s.item_as_boolean (offset + idx + 1) then val_real64 := 1.0 else val_real64 := 0.0 end
						end
					end

					-- Store casted value directly in target tensor
					if attached {ET_STORAGE_REAL_64} l_res.storage as ts then
						ts.put_real_64 (val_real64, i)
					elseif attached {ET_STORAGE_REAL_32} l_res.storage as ts then
						ts.put_real_32 (val_real64.truncated_to_real, i)
					elseif attached {ET_STORAGE_INT_32} l_res.storage as ts then
						ts.put_int_32 (val_real64.truncated_to_integer, i)
					elseif attached {ET_STORAGE_BOOL} l_res.storage as ts then
						ts.put_boolean (val_real64 /= 0.0, i)
					end

					i := i + 1
				end
				Result := l_res
			end
		ensure
			result_valid: Result /= Void
		end

feature -- Output


	out: STRING
		do
			Result := "Tensor " + dtype.out + " (" + shape.count.out + "D)"
		end

invariant
	strides_match_shape: strides.count = shape.count
	shape_consistent_with_storage: shape.is_empty or else calculate_product(shape) <= storage.count
end
