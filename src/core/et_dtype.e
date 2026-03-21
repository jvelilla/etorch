note
	description: "[
		Represents the underlying data type of a Tensor (REAL_64, INTEGER_32, etc.)
		Following PyTorch conventions (e.g. torch.float64, torch.int32).
	]"

deferred class
	ET_DTYPE

inherit
	ANY
		redefine
			is_equal, out
		end

feature -- Status

	is_floating: BOOLEAN deferred end
	is_integer: BOOLEAN deferred end
	byte_size: INTEGER deferred end
	name: STRING deferred end
	
	is_default_float: BOOLEAN
		do Result := False end

feature -- Specific Type Checks

	is_float64: BOOLEAN do Result := False end
	is_float32: BOOLEAN do Result := False end
	is_int64: BOOLEAN do Result := False end
	is_int32: BOOLEAN do Result := False end
	is_bool: BOOLEAN do Result := False end

	type: INTEGER_32 deferred end
			-- Internal representation

feature -- Constants

	type_float64: INTEGER_32 = 1
	type_float32: INTEGER_32 = 2
	type_int64: INTEGER_32 = 3
	type_int32: INTEGER_32 = 4
	type_bool: INTEGER_32 = 5

feature -- Comparison

	is_equal (other: like Current): BOOLEAN
			-- Are `Current` and `other` considered equal?
		do
			Result := type = other.type
		end

feature -- Output

	out: STRING
			-- Name of the dtype
		do
			Result := name
		end

end
