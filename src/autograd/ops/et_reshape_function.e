note
	description: "Autograd function for tensor reshaping."

class
	ET_RESHAPE_FUNCTION

inherit
	ET_FUNCTION
		redefine
			default_create
		end

feature {NONE} -- Initialization

	default_create
		do
			create saved_shape.make_empty
		end

feature -- Forward

	forward_with_params (inputs: ARRAY [ET_VALUE]; a_new_shape: ARRAY [INTEGER_32]): ET_VALUE
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			saved_shape := t1.shape.deep_twin
			
			t_res := t1.reshape_internal (a_new_shape)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
		do
			(create {EXCEPTIONS}).raise ("Use forward_with_params for ET_RESHAPE")
			create Result.make (inputs[1].data)
		end

feature -- Backward

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
		do
			create Result.make_filled (grad_output.reshape (saved_shape), 1, 1)
		end

feature {NONE} -- State

	saved_shape: ARRAY [INTEGER_32]

end
