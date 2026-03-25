note
	description: "[
		A sequential container.
		Modules will be added to it in the order they are passed in the constructor.
		Equivalent to torch.nn.Sequential.
	]"

class
	ET_SEQUENTIAL

inherit
	ET_MODULE

create
	make

feature {NONE} -- Initialization

	make (a_modules: ARRAY [ET_MODULE])
			-- Initialize sequentially with a list of modules.
		do
			create {ARRAYED_LIST [ET_MODULE]} modules.make_from_array (a_modules)
		end

feature -- Access

	modules: LIST [ET_MODULE]

	item alias "[]" (i: INTEGER_32): ET_MODULE
			-- Access the i-th module (1-based index).
		require
			valid_index: i >= 1 and i <= modules.count
		do
			Result := modules.i_th (i)
		end

	parameters: LIST [ET_PARAMETER]
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
			across modules as m loop
				Result.append (m.parameters)
			end
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Sequentially apply each module's forward to the input.
		do
			Result := x
			across modules as m loop
				Result := m.forward (Result)
			end
		end

end
