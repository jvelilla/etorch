note
	description: "Registry for default data types."

class
	ET_DTYPE_REGISTRY

feature -- Access

	default_float_dtype: ET_DTYPE
			-- The current default floating point dtype.
		do
			create {ET_DTYPE_FLOAT32} Result
		ensure
			instance_free: class
		end

end
