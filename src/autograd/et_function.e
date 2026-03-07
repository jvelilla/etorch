note
	description: "[
		Deferred base class for all Autograd operations.
		In PyTorch, every operation over Tensors inherits from `Function`.
	]"

deferred class
	ET_FUNCTION

feature -- Operational

	forward (inputs: ARRAY [ET_VALUE]): ET_VALUE
			-- Computes the forward pass and returns the resulting value.
		require
			valid_inputs: inputs /= Void and then not inputs.is_empty
		deferred
		ensure
			valid_result: Result /= Void
			result_linked: Result.grad_fn = Current
		end

	backward (grad_output: ET_TENSOR): ARRAY [ET_TENSOR]
			-- Computes the gradients with respect to the inputs.
		require
			valid_grad: grad_output /= Void
		deferred
		ensure
			valid_result: Result /= Void
		end

end
