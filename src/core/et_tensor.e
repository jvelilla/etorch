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
	make_ones

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
			create dtype.make_float64
		ensure
			storage_set: storage = a_storage
			offset_set: offset = a_offset
		end

	make_zeros (a_shape: ARRAY [INTEGER_32])
			-- Create a new ZERO-initialized tensor of float64.
		require
			shape_lower_one: a_shape.lower = 1
		local
			l_count: INTEGER_32
			l_store: ET_STORAGE_REAL_64
			l_strides: ARRAY [INTEGER_32]
		do
			shape := a_shape.deep_twin
			l_strides := calculate_contiguous_strides (a_shape)
			strides := l_strides
			offset := 0
			l_count := calculate_product (a_shape)
			
			create l_store.make (l_count)
			storage := l_store
			create device.make_cpu
			create dtype.make_float64
		ensure
			shape_set: shape.count = a_shape.count
		end

	make_ones (a_shape: ARRAY [INTEGER_32])
			-- Create a new ONE-initialized tensor of float64.
		require
			shape_lower_one: a_shape.lower = 1
		local
			l_count: INTEGER_32
			l_store: ET_STORAGE_REAL_64
			l_strides: ARRAY [INTEGER_32]
			i: INTEGER_32
		do
			shape := a_shape.deep_twin
			l_strides := calculate_contiguous_strides (a_shape)
			strides := l_strides
			offset := 0
			l_count := calculate_product (a_shape)
			
			create l_store.make (l_count)
			from i := 1 until i > l_count loop
				l_store.put_real_64 (1.0, i)
				i := i + 1
			end
			storage := l_store
			create device.make_cpu
			create dtype.make_float64
		ensure
			shape_set: shape.count = a_shape.count
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
		
	numel: INTEGER_32
			-- Total number of elements physically representing this shape.
		do
			Result := calculate_product (shape)
		end

feature -- Autograd Props

	grad: detachable ET_TENSOR
			-- Gradient of this tensor.
			
	set_grad (a_grad: ET_TENSOR)
			-- Manually set the gradient (used dynamically or for tests).
		do
			grad := a_grad
		end

	set_requires_grad (val: BOOLEAN)
			-- Enable/disable Autograd tracking.
		do
			requires_grad := val
		end
		
	backward
			-- Trigger backpropagation.
		require
			requires_grad: requires_grad
		do
			-- TODO: Implement Autograd logic here
		end

feature -- Math Operations (with strict Contracts)

	plus alias "+" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise addition.
		require
			same_shape: is_broadcastable (other.shape)
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i: INTEGER_32
		do
			-- Simplified: assume exact same contiguous shape for now.
			-- Real implementation will iter through strides.
			
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			
			from i := 1 until i > l_count loop
				-- Warning: This is a hacky 1D assumption for now, need N-dim strided iter
				l_store.put_real_64 (storage.item_as_real_64 (offset + i) + other.storage.item_as_real_64 (other.offset + i), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	matmul (other: ET_TENSOR): ET_TENSOR
			-- Matrix multiplication (2D for now).
		require
			valid_rank: rank = 2 and other.rank = 2
			compatible_dims: shape [2] = other.shape [1]
		local
			l_m, l_n, l_k: INTEGER_32
			l_lda, l_ldb, l_ldc: INTEGER_32
			l_trans_b: INTEGER_32
			l_res_shape: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_64
			l_res_strides: ARRAY [INTEGER_32]
			l_blas: ET_BLAS
		do
			l_m := shape [1]
			l_k := shape [2]
			l_n := other.shape [2]
			
			l_res_shape := <<l_m, l_n>>
			create l_res_store.make (l_m * l_n)
			l_res_strides := calculate_contiguous_strides (l_res_shape)
			
			create l_blas
			
			-- Detect if `other` is transposed (1D contiguous on dim 1)
			if other.strides [1] = 1 and then other.strides [2] = other.shape [1] then
				l_trans_b := 112 -- CblasTrans
				l_ldb := l_k -- LDB is number of physical columns (k)
			else
				l_trans_b := 111 -- CblasNoTrans
				l_ldb := l_n
			end
			
			l_lda := l_k
			l_ldc := l_n
			
			l_blas.cblas_dgemm (101, 111, l_trans_b, l_m, l_n, l_k, 1.0, storage.data_pointer, l_lda, other.storage.data_pointer, l_ldb, 0.0, l_res_store.data_pointer, l_ldc)
			
			create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)
		ensure
			result_rank: Result.rank = 2
			result_shape: Result.shape [1] = shape [1] and Result.shape [2] = other.shape [2]
		end

	mul alias "*" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise multiplication.
		require
			same_shape: is_broadcastable (other.shape)
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i: INTEGER_32
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			
			from i := 1 until i > l_count loop
				l_store.put_real_64 (storage.item_as_real_64 (offset + i) * other.storage.item_as_real_64 (other.offset + i), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	div alias "/" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise division.
		require
			same_shape: is_broadcastable (other.shape)
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i: INTEGER_32
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			
			from i := 1 until i > l_count loop
				l_store.put_real_64 (storage.item_as_real_64 (offset + i) / other.storage.item_as_real_64 (other.offset + i), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
		end

	exp_val: ET_TENSOR
			-- Element-wise exponential.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i: INTEGER_32
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math
			
			from i := 1 until i > l_count loop
				l_store.put_real_64 (l_math.exp (storage.item_as_real_64 (offset + i)), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	log_val: ET_TENSOR
			-- Element-wise natural logarithm.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i: INTEGER_32
			l_math: DOUBLE_MATH
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math
			
			from i := 1 until i > l_count loop
				l_store.put_real_64 (l_math.log (storage.item_as_real_64 (offset + i)), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	tanh_val: ET_TENSOR
			-- Element-wise hyperbolic tangent.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i: INTEGER_32
			l_math: DOUBLE_MATH
			l_val, l_t, l_e2x: REAL_64
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			create l_math
			
			from i := 1 until i > l_count loop
				l_val := storage.item_as_real_64 (offset + i)
				if l_val > 20.0 then
					l_t := 1.0
				elseif l_val < -20.0 then
					l_t := -1.0
				else
					l_e2x := l_math.exp (2.0 * l_val)
					l_t := (l_e2x - 1.0) / (l_e2x + 1.0)
				end
				l_store.put_real_64 (l_t, i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

feature -- Reductions

	sum (dim: INTEGER_32; keep_dim: BOOLEAN): ET_TENSOR
			-- Sum reduction over a specific dimension.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_res_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_64
			i, k: INTEGER_32
			l_val, l_sum: REAL_64
			l_idx: INTEGER_32
			l_count: INTEGER_32
		do
			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1
			
			l_count := calculate_product (l_res_shape)
			create l_res_store.make (l_count)
			l_res_strides := calculate_contiguous_strides (l_res_shape)
			
			-- Very simplified reduction (assuming contiguous underlying storage for testing)
			-- Proper N-dim stride iterator needed for production
			from i := 1 until i > l_count loop
				l_sum := 0.0
				from k := 1 until k > shape [dim] loop
					-- Calculate index (simplified)
					l_idx := offset + (i - 1) * strides [dim] + (k - 1)
					if l_idx + 1 <= storage.count then
						l_sum := l_sum + storage.item_as_real_64 (l_idx + 1)
					end
					k := k + 1
				end
				l_res_store.put_real_64 (l_sum, i)
				i := i + 1
			end
			
			if not keep_dim then
				-- Remove dimension logic... 
				-- Simplified for test compatibility 
			end
			
			create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)
		ensure
			valid_result: Result /= Void
		end

	mean: ET_TENSOR
			-- Global mean of all elements in the tensor. Returns a scalar tensor.
		local
			l_sum: REAL_64
			i: INTEGER_32
			l_store: ET_STORAGE_REAL_64
		do
			l_sum := 0.0
			from i := 1 until i > numel loop
				l_sum := l_sum + storage.item_as_real_64 (offset + i)
				i := i + 1
			end
			
			create l_store.make (1)
			l_store.put_real_64 (l_sum / numel, 1)
			create Result.make_from_storage (l_store, <<1>>, <<1>>, 0)
		end


feature -- Views (Zero-copy)

	view (new_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Return a new tensor with the same data but different shape.
		require
			view_legal: calculate_product (shape) = calculate_product (new_shape)
		local
			l_strides: ARRAY [INTEGER_32]
		do
			l_strides := calculate_contiguous_strides (new_shape) -- simplified
			create Result.make_from_storage (storage, new_shape, l_strides, offset)
		ensure
			zero_copy: Result.storage = storage
			shape_set: Result.shape ~ new_shape
		end

	reshape (a_new_shape: ARRAY [INTEGER_32]): ET_TENSOR
			-- Like view, but guaranteed to return a tensor with the new shape.
			-- (In a true impl, might copy if non-contiguous, here we assume contiguous views).
		require
			valid_reshape: calculate_product (shape) = calculate_product (a_new_shape)
		do
			Result := view (a_new_shape)
		ensure
			shape_set: Result.shape ~ a_new_shape
		end

	transpose (dim1, dim2: INTEGER_32): ET_TENSOR
			-- Returns a tensor that is a transposed version of `Current`.
			-- (0-based mathematically, adapt internally to 1-based Eiffel structs).
		require
			valid_dim1: dim1 >= 1 and dim1 <= rank
			valid_dim2: dim2 >= 1 and dim2 <= rank
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

feature -- Output

	out: STRING
		do
			Result := "Tensor float64 (" + shape.count.out + "D)"
		end

invariant
	strides_match_shape: strides.count = shape.count
	shape_consistent_with_storage: shape.is_empty or else calculate_product(shape) <= storage.count
end
