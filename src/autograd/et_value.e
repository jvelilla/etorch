note
	description: "[
		Autograd engine node. Wraps an ET_TENSOR to provide gradient tracking.
		Equivalent to PyTorch's `Tensor` when `requires_grad=True` in higher level APIs,
		but kept explicit here for clarity in the engine layer.
	]"

class
	ET_VALUE

create
	make,
	make_with_parents

feature {NONE} -- Initialization

	make (a_data: ET_TENSOR)
			-- Create a leaf node with `a_data`.
		require
			-- No preconditions needed, all params are attached
		do
			data := a_data
			create parents.make_empty
		ensure
			data_set: data = a_data
			no_parents: parents.is_empty
			is_leaf: is_leaf
		end

	make_with_parents (a_data: ET_TENSOR; a_parents: ARRAY [ET_VALUE]; a_fn: ET_FUNCTION)
			-- Create an intermediate node.
		require
			-- No preconditions needed, all params are attached
		do
			data := a_data
			parents := a_parents
			grad_fn := a_fn
		ensure
			data_set: data = a_data
			parents_set: parents = a_parents
			fn_set: grad_fn = a_fn
			not_leaf: not is_leaf
		end

feature -- Access

	data: ET_TENSOR
			-- The actual tensor data.

	grad: detachable ET_TENSOR
			-- Gradient tensor accumulating backward passes.

	grad_fn: detachable ET_FUNCTION
			-- The function that created this value (if not a leaf).

	parents: ARRAY [ET_VALUE]
			-- Values that were used to compute this one.

	is_leaf: BOOLEAN
			-- Is this a leaf node (user created, no grad_fn)?
		do
			Result := grad_fn = Void
		end

feature -- Autograd

	backward
			-- Trigger backpropagation from this node.
		require
			can_backward: is_leaf or else grad_fn /= Void
		do
			-- Auto-grad topology logic will be implemented here
		end

invariant
	leaf_consistency: is_leaf = (grad_fn = Void)
end
