note
	description: "[
		Represents a computing device (CPU, CUDA, NEON, etc.) for tensor operations.
		Following PyTorch conventions.
	]"

class
	ET_DEVICE

create
	make_cpu

feature {NONE} -- Initialization

	make_cpu
			-- Initialize as CPU device.
		do
			type := cpu_type
		ensure
			is_cpu: is_cpu
		end

feature -- Access

	is_cpu: BOOLEAN
			-- Is this device the CPU?
		do
			Result := type = cpu_type
		end

	type: INTEGER_32
			-- Internal representation of the device form

feature -- Constants

	cpu_type: INTEGER_32 = 1
			-- Constant for CPU

invariant
	valid_type: type = cpu_type -- Add more devices here as they are supported (CUDA, etc.)

end
