note
	description: "Bool data type (boolean)."

class
	ET_DTYPE_BOOL

inherit
	ET_DTYPE
		redefine
			is_bool
		end

feature -- Status

	is_floating: BOOLEAN do Result := False end
	is_integer: BOOLEAN do Result := False end
	byte_size: INTEGER do Result := 1 end
	name: STRING do Result := "bool" end

	is_bool: BOOLEAN do Result := True end
	type: INTEGER_32 do Result := type_bool end

end
