note
	description: "Float64 data type (double precision)."

class
	ET_DTYPE_FLOAT64

inherit
	ET_DTYPE
		redefine
			is_float64
		end

feature -- Status

	is_floating: BOOLEAN do Result := True end
	is_integer: BOOLEAN do Result := False end
	byte_size: INTEGER do Result := 8 end
	name: STRING do Result := "float64" end

	is_float64: BOOLEAN do Result := True end
	type: INTEGER_32 do Result := type_float64 end

end
