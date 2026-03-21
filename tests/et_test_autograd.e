note
	description: "Tests for ET_ADAM and autograd tracking."

class
	ET_TEST_AUTOGRAD

inherit
	EQA_TEST_SET

feature -- Tests

	test_adam_step
		local
			p: ET_TENSOR
			g: ET_TENSOR
			optim: ET_ADAM
			params: ARRAYED_LIST [ET_TENSOR]
			tol: REAL_64
		do
			tol := 1.0e-5
			
			create p.make_zeros_with_dtype (<<2>>, create {ET_DTYPE_FLOAT64})
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				ps.put_real_64 (1.0, 1) -- value: 1.0
				ps.put_real_64 (2.0, 2) -- value: 2.0
			end
			p.set_requires_grad (True)
			
			create g.make_zeros_with_dtype (<<2>>, create {ET_DTYPE_FLOAT64})
			if attached {ET_STORAGE_REAL_64} g.storage as gs then
				gs.put_real_64 (0.1, 1) -- grad: 0.1
				gs.put_real_64 (0.2, 2) -- grad: 0.2
			end
			p.set_grad (g)
			
			create params.make (1)
			params.extend (p)
			
			create optim.make (params, 0.1) -- lr = 0.1
			optim.step
			
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				assert_approx_32 (ps.item_as_real_64 (1), 0.9, tol, "P[1] after Adam")
				assert_approx_32 (ps.item_as_real_64 (2), 1.9, tol, "P[2] after Adam")
			else
				assert ("Storage should be real 64", False)
			end
		end

feature {NONE} -- Helpers

	assert_approx_32 (actual, expected, tol: REAL_64; msg: STRING)
		do
			assert (msg + " Expected " + expected.out + " but got " + actual.out, (actual - expected).abs <= tol)
		end

end
