note
	description: "[
		Represents the underlying data type of a Tensor (REAL_64, INTEGER_32, etc.)
		Following PyTorch conventions (e.g. torch.float64, torch.int32).
	]"

class
	ET_DTYPE

inherit
	ANY
		redefine
			is_equal,
			out
		end

create
	make_float64,
	make_float32,
	make_int64,
	make_int32,
	make_bool

feature {NONE} -- Initialization

	make_float64
		do
			type := type_float64
		ensure
			is_float64: is_float64
		end

	make_float32
		do
			type := type_float32
		ensure
			is_float32: is_float32
		end

	make_int64
		do
			type := type_int64
		ensure
			is_int64: is_int64
		end

	make_int32
		do
			type := type_int32
		ensure
			is_int32: is_int32
		end

	make_bool
		do
			type := type_bool
		ensure
			is_bool: is_bool
		end

feature -- Access

	is_float64: BOOLEAN do Result := type = type_float64 end
	is_float32: BOOLEAN do Result := type = type_float32 end
	is_int64: BOOLEAN do Result := type = type_int64 end
	is_int32: BOOLEAN do Result := type = type_int32 end
	is_bool: BOOLEAN do Result := type = type_bool end

	type: INTEGER_32
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
			if is_float64 then
				Result := "float64"
			elseif is_float32 then
				Result := "float32"
			elseif is_int64 then
				Result := "int64"
			elseif is_int32 then
				Result := "int32"
			elseif is_bool then
				Result := "bool"
			else
				Result := "unknown"
			end
		end

invariant
	valid_type: type >= type_float64 and type <= type_bool

end
