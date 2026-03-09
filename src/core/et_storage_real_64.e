note
	description: "[
		Concrete storage for REAL_64 (float64) data.
		Wraps a standard Eiffel ARRAY [REAL_64] for safety and interoperability.
	]"

class
	ET_STORAGE_REAL_64

inherit
	ET_STORAGE

create
	make,
	make_from_array

feature {NONE} -- Initialization

	make (n: INTEGER_32)
			-- Create a zero-initialized storage of size `n`.
		require
			n_positive: n >= 0
		do
			create data.make_filled (0.0, 1, n)
		ensure
			count_set: count = n
		end

	make_from_array (arr: ARRAY [REAL_64])
			-- Create storage pointing to the existing `arr`.
		require
			arr_starts_at_1: arr.lower = 1
		do
			data := arr
		ensure
			data_set: data = arr
		end

feature -- Access

	count: INTEGER_32
			-- Number of elements in the storage.
		do
			Result := data.count
		end

	item_as_real_64 (index: INTEGER_32): REAL_64
			-- Read an element as REAL_64.
		do
			Result := data.item (index)
		end

	put_real_64 (v: REAL_64; index: INTEGER_32)
			-- Write an element as REAL_64.
		do
			data.put (v, index)
		end

feature {ET_TENSOR} -- Internal Data

	data: ARRAY [REAL_64]
			-- The underlying array

feature -- C Interoperability

	data_pointer: POINTER
			-- Raw pointer to the underlying storage memory.
		do
			Result := data.area.base_address
		end

invariant
	lower_is_one: data.lower = 1
end
