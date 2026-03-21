note
	description: "Int32 data type (32-bit integer)."

class
	ET_DTYPE_INT32

inherit
	ET_DTYPE
		redefine
			is_int32
		end

feature -- Status

	is_floating: BOOLEAN do Result := False end
	is_integer: BOOLEAN do Result := True end
	byte_size: INTEGER do Result := 4 end
	name: STRING do Result := "int32" end

	is_int32: BOOLEAN do Result := True end
	type: INTEGER_32 do Result := type_int32 end

end
