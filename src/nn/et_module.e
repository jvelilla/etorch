note
	description: "[
		Base class for all neural network modules.
		Your models should also inherit from this class.
	]"

deferred class
	ET_MODULE

feature -- Access

	parameters: LIST [ET_PARAMETER]
			-- Returns an iterator over module parameters.
			-- This is typically overriden by concrete modules or collected via reflection in advanced implementations.
		deferred
		end

	state_dict: HASH_TABLE [ET_TENSOR, STRING]
			-- Returns a dictionary containing a whole state of the module.
		do
			create Result.make (0)
		end

feature -- Mode Configuration

	train
			-- Sets the module in training mode.
		do
			is_training := True
		ensure
			training_mode: is_training
		end

	eval
			-- Sets the module in evaluation mode.
		do
			is_training := False
		ensure
			eval_mode: not is_training
		end

	is_training: BOOLEAN
			-- Is the module in training mode?

feature -- Core Operation

	forward (x: ET_TENSOR): ET_TENSOR
			-- Defines the computation performed at every call.
			-- Should be overridden by all subclasses.
		deferred
		ensure
			valid_output: Result /= Void
		end

feature -- Utilities

	zero_grad
			-- Sets gradients of all model parameters to zero.
		do
			across parameters as p loop
				-- Equivalent of p.grad = None in PyTorch
				if attached p.grad as g then
					-- g.zero_() logic
				end
			end
		end

	to_device (d: ET_DEVICE)
			-- Moves all parameters to the specified device.
		do
			-- In v2, this would iterate over parameters and move underlying storage
		end

	load_state_dict (state: HASH_TABLE [ET_TENSOR, STRING])
			-- Copies parameters and buffers from `state` into this module and its descendants.
		do
			-- In v2, this applies the tensors manually
		end

end
