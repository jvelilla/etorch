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
		local
			l_val: ET_VALUE
		do
			create l_val.make (Current)
			l_val.backward
		end

feature -- Math Operations (with strict Contracts)

	plus alias "+" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise addition.
		require
			same_shape: is_broadcastable (other.shape)
		local
			l_strides, br_shape: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self, idx_other: INTEGER_32
		do
			br_shape := broadcast_shape (shape, other.shape)
			l_count := calculate_product (br_shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (br_shape)
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, br_shape)
				idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
				l_store.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) + other.storage.item_as_real_64 (other.offset + idx_other + 1), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, br_shape, l_strides, 0)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
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
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, br_shape)
				idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
				storage.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) + other.storage.item_as_real_64 (other.offset + idx_other + 1), offset + idx_self + 1)
				i := i + 1
			end
		end

	plus_scalar (val: REAL_64): ET_TENSOR
			-- Element-wise addition with a scalar.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_store.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) + val, i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	plus_scalar_in_place (val: REAL_64)
			-- In-place element-wise addition with a scalar.
		require
			not_requires_grad: not requires_grad
		local
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				storage.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) + val, offset + idx_self + 1)
				i := i + 1
			end
		end

	matmul (other: ET_TENSOR): ET_TENSOR
			-- Matrix multiplication (supports 1D dot, 2D matmul, and N-D batched matmul).
		require
			valid_operands: rank >= 1 and other.rank >= 1
		local
			l_res_shape: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_64
			l_res_strides: ARRAY [INTEGER_32]
			l_m, l_n, l_k: INTEGER_32
			l_lda, l_ldb, l_ldc: INTEGER_32
			l_trans_a, l_trans_b: INTEGER_32
			l_blas: ET_BLAS
			i, l_trans_a_iter: INTEGER_32
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
					from l_trans_a_iter := 0 until l_trans_a_iter >= l_k loop
						l_dot := l_dot + storage.item_as_real_64(offset + idx_a_batch + l_trans_a_iter * strides[rank] + 1) * other.storage.item_as_real_64(other.offset + l_trans_a_iter * other.strides[1] + 1)
						l_trans_a_iter := l_trans_a_iter + 1
					end
					l_res_store.put_real_64(l_dot, i)
					i := i + 1
				end
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)
				
			else
				-- N-D Batched Matmul / 2D Matmul
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
				if l_batch_count = 0 then l_batch_count := 1 end -- for 2D where batch shape is empty
				
				create l_res_store.make (l_batch_count * l_m * l_n)
				l_res_strides := calculate_contiguous_strides (l_res_shape)
				
				create l_blas
				
				if strides.count >= 2 and then strides [rank-1] = 1 and then strides [rank] = l_m then
					l_trans_a := 112 -- CblasTrans
					l_lda := l_m
				else
					l_trans_a := 111 -- CblasNoTrans
					l_lda := l_k
				end
				
				if other.strides.count >= 2 and then other.strides [other.rank-1] = 1 and then other.strides [other.rank] = l_k then
					l_trans_b := 112 -- CblasTrans
					l_ldb := l_k
				else
					l_trans_b := 111 -- CblasNoTrans
					l_ldb := l_n
				end
				
				l_ldc := l_n
				
				if res_batch_shape.count = 0 then
					l_blas.cblas_dgemm (101, l_trans_a, l_trans_b, l_m, l_n, l_k, 1.0, 
						storage.data_pointer + offset * 8, l_lda, 
						other.storage.data_pointer + other.offset * 8, l_ldb, 
						0.0, l_res_store.data_pointer, l_ldc)
				else
					from i := 1 until i > l_batch_count loop
						idx_a_batch := linear_index_to_offset(i, a_batch_shape, strides, res_batch_shape)
						idx_b_batch := linear_index_to_offset(i, b_batch_shape, other.strides, res_batch_shape)
						offset_c := (i - 1) * l_m * l_n
						
						l_blas.cblas_dgemm (101, l_trans_a, l_trans_b, l_m, l_n, l_k, 1.0, 
							storage.data_pointer + (offset + idx_a_batch) * 8, l_lda, 
							other.storage.data_pointer + (other.offset + idx_b_batch) * 8, l_ldb, 
							0.0, l_res_store.data_pointer + offset_c * 8, l_ldc)
						i := i + 1
					end
				end
				
				create Result.make_from_storage (l_res_store, l_res_shape, l_res_strides, 0)
			end
		end

	mul alias "*" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise multiplication.
		require
			same_shape: is_broadcastable (other.shape)
		local
			l_strides, br_shape: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self, idx_other: INTEGER_32
		do
			br_shape := broadcast_shape (shape, other.shape)
			l_count := calculate_product (br_shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (br_shape)
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, br_shape)
				idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
				l_store.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) * other.storage.item_as_real_64 (other.offset + idx_other + 1), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, br_shape, l_strides, 0)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
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
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, br_shape)
				idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
				storage.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) * other.storage.item_as_real_64 (other.offset + idx_other + 1), offset + idx_self + 1)
				i := i + 1
			end
		end

	mul_scalar (val: REAL_64): ET_TENSOR
			-- Element-wise multiplication with a scalar.
		local
			l_strides: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (shape)
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				l_store.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) * val, i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

	mul_scalar_in_place (val: REAL_64)
			-- In-place element-wise multiplication with a scalar.
		require
			not_requires_grad: not requires_grad
		local
			l_count, i, idx_self: INTEGER_32
		do
			l_count := calculate_product (shape)
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, shape)
				storage.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) * val, offset + idx_self + 1)
				i := i + 1
			end
		end

	div alias "/" (other: ET_TENSOR): ET_TENSOR
			-- Element-wise division.
		require
			same_shape: is_broadcastable (other.shape)
		local
			l_strides, br_shape: ARRAY [INTEGER_32]
			l_store: ET_STORAGE_REAL_64
			l_count, i, idx_self, idx_other: INTEGER_32
		do
			br_shape := broadcast_shape (shape, other.shape)
			l_count := calculate_product (br_shape)
			create l_store.make (l_count)
			l_strides := calculate_contiguous_strides (br_shape)
			
			from i := 1 until i > l_count loop
				idx_self := linear_index_to_offset (i, shape, strides, br_shape)
				idx_other := linear_index_to_offset (i, other.shape, other.strides, br_shape)
				l_store.put_real_64 (storage.item_as_real_64 (offset + idx_self + 1) / other.storage.item_as_real_64 (other.offset + idx_other + 1), i)
				i := i + 1
			end
			
			create Result.make_from_storage (l_store, br_shape, l_strides, 0)
		ensure
			result_shape_correct: Result.shape ~ broadcast_shape (shape, other.shape)
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
			
			create Result.make_from_storage (l_store, shape, l_strides, 0)
		end

feature -- Reductions

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
			l_res_shape, l_final_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_64
			i, k, idx_src: INTEGER_32
			l_sum: REAL_64
			l_count: INTEGER_32
		do
			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1
			
			l_count := calculate_product (l_res_shape)
			create l_res_store.make (l_count)
			
			from i := 1 until i > l_count loop
				idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
				l_sum := 0.0
				from k := 0 until k >= shape [dim] loop
					l_sum := l_sum + storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
					k := k + 1
				end
				l_res_store.put_real_64 (l_sum, i)
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
				if l_final_shape.is_empty then
					l_final_shape.force (1, 1)
				end
			end
			
			l_res_strides := calculate_contiguous_strides (l_final_shape)
			create Result.make_from_storage (l_res_store, l_final_shape, l_res_strides, 0)
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

	max_dim (dim: INTEGER_32; keep_dim: BOOLEAN): ET_TENSOR
			-- Max reduction over a specific dimension.
		require
			valid_dim: dim >= 1 and dim <= rank
		local
			l_res_shape, l_final_shape: ARRAY [INTEGER_32]
			l_res_strides: ARRAY [INTEGER_32]
			l_res_store: ET_STORAGE_REAL_64
			i, k, idx_src: INTEGER_32
			l_max, l_val: REAL_64
			l_count: INTEGER_32
		do
			l_res_shape := shape.deep_twin
			l_res_shape [dim] := 1
			
			l_count := calculate_product (l_res_shape)
			create l_res_store.make (l_count)
			
			from i := 1 until i > l_count loop
				idx_src := linear_index_to_offset (i, shape, strides, l_res_shape)
				l_max := -1.7976931348623157e+308
				from k := 0 until k >= shape [dim] loop
					l_val := storage.item_as_real_64 (offset + idx_src + k * strides [dim] + 1)
					if l_val > l_max or k = 0 then l_max := l_val end
					k := k + 1
				end
				l_res_store.put_real_64 (l_max, i)
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
			create Result.make_from_storage (l_res_store, l_final_shape, l_res_strides, 0)
		end

	softmax (dim: INTEGER_32): ET_TENSOR
			-- Softmax over dimension `dim`.
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
