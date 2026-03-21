note
	description: "Singleton registry for default data types."

class
	ET_DTYPE_REGISTRY

feature -- Access

	default_float_dtype: ET_DTYPE
			-- The current default floating point dtype.
		do
			Result := default_float_cell.item
		end

	set_default_float_dtype (a_dtype: ET_DTYPE)
			-- Set the default floating point dtype.
		require
			valid_floating_type: a_dtype.is_floating
		do
			default_float_cell.put (a_dtype)
		ensure
			default_updated: default_float_dtype = a_dtype
		end

feature {NONE} -- Implementation

	default_float_cell: CELL [ET_DTYPE]
			-- Storage for the default dtype.
		once
			create Result.put (create {ET_DTYPE_FLOAT32})
		end

end
