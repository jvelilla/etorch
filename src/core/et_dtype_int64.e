note
	description: "Int64 data type (64-bit integer)."

class
	ET_DTYPE_INT64

inherit
	ET_DTYPE
		redefine
			is_int64
		end

feature -- Status

	is_floating: BOOLEAN do Result := False end
	is_integer: BOOLEAN do Result := True end
	byte_size: INTEGER do Result := 8 end
	name: STRING do Result := "int64" end

	is_int64: BOOLEAN do Result := True end
	type: INTEGER_32 do Result := type_int64 end

end
