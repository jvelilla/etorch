note
	description: "End-to-End Training Autograd Tests"

class
	ET_TEST_TRAINING

inherit
	EQA_TEST_SET
		redefine
			on_prepare
		select
			default_create
		end
	DOUBLE_MATH
		rename
			default_create as dm_default_create
		end

feature -- Initialization

	on_prepare
			-- Called before each test.
		do
		end

feature -- Tests

	test_tensor_autograd
		local
			x, y, z: ET_TENSOR
			tol: REAL_64
		do
			print ("  [TEST] Tensor Autograd Basic Add/Mul... ")
			tol := 1.0e-5
			
			create x.make_ones (<<1>>)
			x := x.mul_scalar (2.0)
			x.set_requires_grad (True)
			
			create y.make_ones (<<1>>)
			y := y.mul_scalar (3.0)
			y.set_requires_grad (True)
			
			-- z = x * y + y
			-- dz/dx = y = 3.0
			-- dz/dy = x + 1 = 2.0 + 1 = 3.0
			z := x.mul (y).plus (y)
			z.backward
			
			if attached x.grad as g_x then
				assert_approx (g_x.item_as_real_64 (<<1>>), 3.0, tol, "dz/dx should be 3.0")
			else
				assert ("x.grad is not null", False)
			end
			
			if attached y.grad as g_y then
				assert_approx (g_y.item_as_real_64 (<<1>>), 3.0, tol, "dz/dy should be 3.0")
			else
				assert ("y.grad is not null", False)
			end
			
			print ("OK%N")
		end

	test_attention_autograd
		local
			q, k, v, att, att_probs, l_out, loss: ET_TENSOR
			tol: REAL_64
			t_scale: ET_TENSOR
			b, t, c_: INTEGER
			causal_mask: ET_TENSOR
			t1, t2: INTEGER
			scalar_shape: ARRAY [INTEGER]
		do
			print ("  [TEST] Attention Autograd... ")
			tol := 1.0e-4
			
			b := 2
			t := 3
			c_ := 4
			
			create q.make_ones_with_dtype (<<b, t, c_>>, create {ET_DTYPE_FLOAT64})
			create k.make_ones_with_dtype (<<b, t, c_>>, create {ET_DTYPE_FLOAT64})
			create v.make_ones_with_dtype (<<b, t, c_>>, create {ET_DTYPE_FLOAT64})
			
			q.set_requires_grad (True)
			k.set_requires_grad (True)
			v.set_requires_grad (True)
			
			att := q.matmul (k.transpose (2, 3))
			
			create scalar_shape.make_empty
			create t_scale.make_ones_with_dtype (scalar_shape, create {ET_DTYPE_FLOAT64})
			t_scale := t_scale.mul_scalar (1.0 / c_.to_double.power (0.5))
			att := att.mul (t_scale)
			
			create causal_mask.make_zeros_with_dtype (<<t, t>>, create {ET_DTYPE_FLOAT64})
			
			from t1 := 1 until t1 > t loop
				from t2 := t1 + 1 until t2 > t loop
					causal_mask.put_real_64 (-1.0e9, <<t1, t2>>)
					t2 := t2 + 1
				end
				t1 := t1 + 1
			end
			
			att := att.plus (causal_mask)
			
			-- Testing Softmax activation backwards
			att_probs := att.softmax (3)
			
			-- Testing Matmul backward
			l_out := att_probs.matmul (v)
			
			-- Emulate loss over multiple sum reductions
			loss := l_out.sum (<<1, 2, 3>>, False)
			loss.backward
			
			if attached q.grad as g_q then
				assert_approx (g_q.mean_all.item_as_real_64 (<<1>>), 0.0, 1.0, "q grad mean")
			else
				assert ("q.grad is not null", False)
			end
			
			if attached v.grad as g_v then
				assert_approx (g_v.mean_all.item_as_real_64 (<<1>>), 1.0, 1.0e-3, "v grad mean")
			else
				assert ("v.grad is not null", False)
			end
			
			print ("OK%N")
		end

	test_mlp_training_step
		local
			linear1, linear2: ET_LINEAR
			adam: ET_ADAM
			x, target, logits, loss, diff, diff_sq: ET_TENSOR
			params1, params2: LIST [ET_PARAMETER]
			optimizer_params: ARRAYED_LIST [ET_TENSOR]
		do
			print ("  [TEST] MLP End-to-End Training Step... ")
			
			-- Models setup
			create linear1.make (2, 4)
			create linear2.make (4, 1)
			
			-- Input and target
			create x.make_randn (<<2>>)
			create target.make_ones (<<1>>)
			
			params1 := linear1.parameters
			params2 := linear2.parameters
			
			create optimizer_params.make (params1.count + params2.count)
			from params1.start until params1.after loop
				optimizer_params.extend (params1.item)
				params1.forth
			end
			from params2.start until params2.after loop
				optimizer_params.extend (params2.item)
				params2.forth
			end
			
			-- Optimizer setup
			create adam.make (optimizer_params, 0.1)
			
			-- 1. Forward Pass
			logits := linear1.forward (x).gelu
			logits := linear2.forward (logits)
			
			-- Mean Squared Error Loss
			diff := logits.minus (target)
			diff_sq := diff.mul (diff)
			loss := diff_sq.mean_all
			
			-- 2. Backward Pass
			loss.backward
			
			assert ("linear2 weight grad attached", linear2.weight.grad /= Void)
			assert ("linear1 weight grad attached", linear1.weight.grad /= Void)
			
			-- 3. Optimizer Step
			adam.step
			adam.zero_grad
			
			assert ("linear2 weight grad zeroed", linear2.weight.grad = Void)
			
			print ("OK%N")
		end

	test_relu_autograd
			-- Verify ReLU forward + backward: mask is 0 for x<=0, 1 for x>0.
		local
			x, y: ET_TENSOR
			tol: REAL_64
		do
			print ("  [TEST] ReLU Autograd... ")
			tol := 1.0e-5

				-- x = [-1, 0, 2]  with requires_grad
			create x.make_zeros_with_dtype (<<3>>, create {ET_DTYPE_FLOAT64})
			if attached {ET_STORAGE_REAL_64} x.storage as s then
				s.put_real_64 (-1.0, 1)
				s.put_real_64 (0.0,  2)
				s.put_real_64 (2.0,  3)
			else
				assert("Storage is ET_STORAGE_REAL_64", False)
			end
			x.set_requires_grad (True)

				-- Forward
			y := x.relu

				-- Backward with all-ones upstream gradient
			y.backward

				-- d(relu)/dx should be: [0, 0, 1]
			if attached x.grad as g then
				assert_approx (g.item_as_real_64 (<<1>>), 0.0, tol, "relu grad[1] (x=-1) should be 0")
				assert_approx (g.item_as_real_64 (<<2>>), 0.0, tol, "relu grad[2] (x=0)  should be 0")
				assert_approx (g.item_as_real_64 (<<3>>), 1.0, tol, "relu grad[3] (x=2)  should be 1")
			else
				assert ("x.grad is not null", False)
			end

			print ("OK%N")
		end

	assert_approx (actual: REAL_64; expected: REAL_64; tol: REAL_64; msg: STRING)
		do
			assert (msg + " Expected " + expected.out + " but got " + actual.out, (actual - expected).abs <= tol)
		end

end
