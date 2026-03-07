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
			storage_attached: a_storage /= Void
			shape_attached: a_shape /= Void and then a_shape.lower = 1
			strides_attached: a_strides /= Void and then a_strides.count = a_shape.count
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
			shape_attached: a_shape /= Void and then a_shape.lower = 1
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
			shape_attached: a_shape /= Void and then a_shape.lower = 1
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
		do
			-- Simplified: only allow exact shapes for now.
			Result := shape ~ other_shape
		end
		
	broadcast_shape (shape1, shape2: ARRAY [INTEGER_32]): ARRAY [INTEGER_32]
		do
			Result := shape1 -- simplified
		end

feature -- Output

	out: STRING
		do
			Result := "Tensor float64 (" + shape.count.out + "D)"
		end

invariant
	-- Core tensors MUST be correct internally by definition
	strides_match_shape: strides.count = shape.count
	device_attached: device /= Void
	storage_attached: storage /= Void
	dtype_attached: dtype /= Void
end
