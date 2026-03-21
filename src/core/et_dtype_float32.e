note
	description: "Float32 data type (single precision)."

class
	ET_DTYPE_FLOAT32

inherit
	ET_DTYPE
		redefine
			is_default_float, is_float32
		end

feature -- Status

	is_floating: BOOLEAN do Result := True end
	is_integer: BOOLEAN do Result := False end
	byte_size: INTEGER do Result := 4 end
	name: STRING do Result := "float32" end
	is_default_float: BOOLEAN do Result := True end

	is_float32: BOOLEAN do Result := True end
	type: INTEGER_32 do Result := type_float32 end

end
