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
		require
			valid_modules: a_modules /= Void
		do
			create {ARRAYED_LIST [ET_MODULE]} modules.make_from_array (a_modules)
		end

feature -- Access

	modules: LIST [ET_MODULE]

	parameters: LIST [ET_PARAMETER]
		do
			create {ARRAYED_LIST [ET_PARAMETER]} Result.make (0)
			across modules as m loop
				Result.append (m.item.parameters)
			end
		end

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Sequentially apply each module's forward to the input.
		do
			Result := x
			across modules as m loop
				Result := m.item.forward (Result)
			end
		end

end
