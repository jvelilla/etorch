note
	description: "[
		Abstract storage layer for a Tensor. It holds the 1D, contiguous raw data.
		A Tensor is just a "view" (shape/strides) over this storage.
	]"

deferred class
	ET_STORAGE

feature -- Access

	count: INTEGER_32
			-- Number of elements in the storage.
		deferred
		ensure
			count_non_negative: Result >= 0
		end

	item_as_real_64 (index: INTEGER_32): REAL_64
			-- Read an element as REAL_64.
		require
			valid_index: valid_index (index)
		deferred
		end

	item_as_real_32 (index: INTEGER_32): REAL_32
			-- Read an element as REAL_32.
		require
			valid_index: valid_index (index)
		deferred
		end

	item_as_int_64 (index: INTEGER_32): INTEGER_64
			-- Read an element as INTEGER_64.
		require
			valid_index: valid_index (index)
		deferred
		end

	item_as_int_32 (index: INTEGER_32): INTEGER_32
			-- Read an element as INTEGER_32.
		require
			valid_index: valid_index (index)
		deferred
		end

	item_as_boolean (index: INTEGER_32): BOOLEAN
			-- Read an element as BOOLEAN.
		require
			valid_index: valid_index (index)
		deferred
		end

	put_real_64 (v: REAL_64; index: INTEGER_32)
			-- Write an element as REAL_64.
		require
			valid_index: valid_index (index)
		deferred
		ensure
			value_set: item_as_real_64 (index) = v
		end

	put_real_32 (v: REAL_32; index: INTEGER_32)
			-- Write an element as REAL_32.
		require
			valid_index: valid_index (index)
		deferred
		ensure
			value_set: item_as_real_32 (index) = v
		end

	put_int_64 (v: INTEGER_64; index: INTEGER_32)
			-- Write an element as INTEGER_64.
		require
			valid_index: valid_index (index)
		deferred
		ensure
			value_set: item_as_int_64 (index) = v
		end

	put_int_32 (v: INTEGER_32; index: INTEGER_32)
			-- Write an element as INTEGER_32.
		require
			valid_index: valid_index (index)
		deferred
		ensure
			value_set: item_as_int_32 (index) = v
		end

	put_boolean (v: BOOLEAN; index: INTEGER_32)
			-- Write an element as BOOLEAN.
		require
			valid_index: valid_index (index)
		deferred
		ensure
			value_set: item_as_boolean (index) = v
		end

feature -- Query

	valid_index (index: INTEGER_32): BOOLEAN
			-- Is `index` valid for this storage? (1-based)
		do
			Result := index >= 1 and index <= count
		ensure
			correct_range: Result = (index >= 1 and index <= count)
		end

feature -- C Interoperability

	data_pointer: POINTER
			-- Raw pointer to the underlying storage memory.
		deferred
		end

invariant
	count_valid: count >= 0
end
