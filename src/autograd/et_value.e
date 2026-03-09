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

	visited: BOOLEAN
			-- Property for topological sorting

	set_visited (v: BOOLEAN)
		do
			visited := v
		end

	set_grad (a_grad: ET_TENSOR)
		do
			grad := a_grad
			if data.requires_grad then
				data.set_grad (a_grad)
			end
		end

feature -- Autograd

	backward
			-- Trigger backpropagation from this node.
		require
			can_backward: is_leaf or else grad_fn /= Void
		local
			topo: ARRAYED_LIST [ET_VALUE]
			i, k: INTEGER_32
			l_one: ET_TENSOR
			grads: ARRAY [ET_TENSOR]
			p: ET_VALUE
		do
			create topo.make (100)
			build_topo (Current, topo)
			
			if grad = Void then
				-- gradient = 1.0 for the scalar leaf if not manually seeded
				create l_one.make_ones (data.shape)
				set_grad (l_one)
			end

			-- Apply backward in reverse topological order
			from i := topo.count until i < 1 loop
				if attached topo [i].grad_fn as gf and then attached topo [i].grad as g then
					grads := gf.backward (g)
					from k := 1 until k > topo [i].parents.count loop
						p := topo [i].parents [k]
						if attached p.grad as cur_grad then
							p.set_grad (cur_grad + grads [k])
						else
							p.set_grad (grads [k])
						end
						k := k + 1
					end
				end
				topo [i].set_visited (False)
				i := i - 1
			end
		end

feature {NONE} -- Helpers

	build_topo (v: ET_VALUE; topo: ARRAYED_LIST [ET_VALUE])
		do
			if not v.visited then
				v.set_visited (True)
				if not v.parents.is_empty then
					across v.parents as child loop
						build_topo (child, topo)
					end
				end
				topo.extend (v)
			end
		end

invariant
	leaf_consistency: is_leaf = (grad_fn = Void)
end
