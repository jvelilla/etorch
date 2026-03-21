note
	description: "Autograd function for tensor transposition."

class
	ET_TRANSPOSE_FUNCTION

inherit
	ET_FUNCTION
		redefine
			default_create
		end

feature {NONE} -- Initialization

	default_create
		do
			saved_dim1 := 1
			saved_dim2 := 1
		end

feature -- Forward

	forward_with_params (inputs: ARRAY [ET_VALUE]; dim1, dim2: INTEGER_32): ET_VALUE
		local
			t1, t_res: ET_TENSOR
		do
			t1 := inputs [1].data
			saved_dim1 := dim1
			saved_dim2 := dim2
			
			t_res := t1.transpose_internal (dim1, dim2)
			create Result.make_with_parents (t_res, inputs, Current)
		end

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
		do
			(create {EXCEPTIONS}).raise ("Use forward_with_params for ET_TRANSPOSE")
			create Result.make (inputs[1].data)
		end

feature -- Backward

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
		do
			create Result.make_filled (grad_output.transpose (saved_dim1, saved_dim2), 1, 1)
		end

feature {NONE} -- State

	saved_dim1, saved_dim2: INTEGER_32

end
