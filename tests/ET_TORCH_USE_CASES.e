note
    description: "PyTorch Use Cases implemented in Eiffel for eTorch."

class
    ET_TORCH_USE_CASES

create
	make

feature -- Initialization

	make
		do
			print ("eTorch Compilation Successful! Tests will be run here.%N")
			from_existing_data
			test_matmul
			test_adam
			test_save_load
			test_autograd_chain
			test_training_loop
			test_math
			test_delivery_prediction
		end

feature -- Tests

    from_existing_data
            -- 1.1 From Existing Data Structures
        local
            x: ET_TENSOR
            l_data: ARRAY [INTEGER_32]
            l_shape: ARRAY [INTEGER_32]
        do
            print ("%N[1.1] From Existing Data Structures ( List -> Tensor)%N")
            
            -- Simplified for compile check - normally we parse the array
            l_data := <<1, 2, 3>>
            l_shape := <<3>>
            create x.make_zeros (l_shape)

            print ("TENSOR NUMEL: " + x.numel.out + "%N")
        end

	test_matmul
		local
			a, b, c: ET_TENSOR
			a_store, b_store: ET_STORAGE_REAL_64
			l_strides: ARRAY [INTEGER_32]
		do
			print ("%N[1.2] Matrix Multiplication (OpenBLAS cblas_dgemm)%N")
			
			create a_store.make (4)
			a_store.put_real_64 (1.0, 1)
			a_store.put_real_64 (2.0, 2)
			a_store.put_real_64 (3.0, 3)
			a_store.put_real_64 (4.0, 4)
			
			create b_store.make (4)
			b_store.put_real_64 (2.0, 1)
			b_store.put_real_64 (0.0, 2)
			b_store.put_real_64 (1.0, 3)
			b_store.put_real_64 (2.0, 4)
			
			l_strides := <<2, 1>>
			
			create a.make_from_storage (a_store, <<2, 2>>, l_strides, 0)
			create b.make_from_storage (b_store, <<2, 2>>, l_strides, 0)
			
			c := a.matmul (b)
			print ("MATMUL Result shape: [" + c.shape [1].out + ", " + c.shape [2].out + "]%N")
			
			print ("C[1,1]: " + c.storage.item_as_real_64 (1).out + " (Expected 4)%N")
			print ("C[1,2]: " + c.storage.item_as_real_64 (2).out + " (Expected 4)%N")
			print ("C[2,1]: " + c.storage.item_as_real_64 (3).out + " (Expected 10)%N")
			print ("C[2,2]: " + c.storage.item_as_real_64 (4).out + " (Expected 8)%N")
		end

	test_adam
		local
			p: ET_TENSOR
			g: ET_TENSOR
			optim: ET_ADAM
			params: ARRAYED_LIST [ET_TENSOR]
		do
			print ("%N[1.3] Adam Optimization Step%N")
			
			create p.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				ps.put_real_64 (1.0, 1) -- value: 1.0
				ps.put_real_64 (2.0, 2) -- value: 2.0
			end
			p.set_requires_grad (True)
			
			create g.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} g.storage as gs then
				gs.put_real_64 (0.1, 1) -- grad: 0.1
				gs.put_real_64 (0.2, 2) -- grad: 0.2
			end
			p.set_grad (g)
			
			create params.make (1)
			params.extend (p)
			
			create optim.make (params, 0.1) -- lr = 0.1
			optim.step
			
			print ("After Adam Step (with LR=0.1):%N")
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				print ("P[1]: " + ps.item_as_real_64 (1).out + "%N")
				print ("P[2]: " + ps.item_as_real_64 (2).out + "%N")
			end
		end

	test_save_load
		local
			p: ET_TENSOR
			state: HASH_TABLE [ET_TENSOR, STRING]
			sl: ET_SAVE_LOAD
			loaded_state: detachable HASH_TABLE [ET_TENSOR, STRING]
		do
			print ("%N[1.4] Model Serialization%N")
			
			create p.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} p.storage as ps then
				ps.put_real_64 (5.5, 1)
				ps.put_real_64 (7.7, 2)
			end
			
			create state.make (1)
			state.put (p, "layer.weight")
			
			create sl
			sl.save (state, "test_model.pt")
			
			loaded_state := sl.load ("test_model.pt")
			if attached loaded_state as ls and then attached ls.item ("layer.weight") as lp then
				print ("Loaded state value P[1]: ")
				if attached {ET_STORAGE_REAL_64} lp.storage as lps then
					print (lps.item_as_real_64 (1).out + " (Expected 5.5)%N")
				end
			else
				print ("Failed to load state!%N")
			end
		end

	test_autograd_chain
		local
			t1, t2: ET_TENSOR
			v1, v2, v_out: ET_VALUE
			l_add: ET_ADD_FUNCTION
			inputs: ARRAY [ET_VALUE]
			g_out: ET_TENSOR
		do
			print ("%N[1.5] Autograd Forward and Backward Pass%N")
			
			create t1.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} t1.storage as s then s.put_real_64 (2.0, 1); s.put_real_64 (3.0, 2) end
			t1.set_requires_grad (True)
			
			create t2.make_zeros (<<2>>)
			if attached {ET_STORAGE_REAL_64} t2.storage as s then s.put_real_64 (4.0, 1); s.put_real_64 (5.0, 2) end
			t2.set_requires_grad (True)
			
			create v1.make (t1)
			create v2.make (t2)
			
			create l_add
			inputs := <<v1, v2>>
			v_out := l_add.forward (inputs)
			
			create g_out.make_ones (v_out.data.shape)
			v_out.set_grad (g_out)
			
			v_out.backward
			
			if attached v1.grad as g1 and attached v2.grad as g2 then
				print ("V1 Grad: [" + g1.storage.item_as_real_64 (1).out + ", " + g1.storage.item_as_real_64 (2).out + "] (Expected [1, 1])%N")
				print ("V2 Grad: [" + g2.storage.item_as_real_64 (1).out + ", " + g2.storage.item_as_real_64 (2).out + "] (Expected [1, 1])%N")
			end
		end

	test_training_loop
		local
			x, target, w, b: ET_TENSOR
			v_x, v_w, v_b: ET_VALUE
			v_mul, v_pred: ET_VALUE
			l_mul: ET_MUL_FUNCTION
			l_add: ET_ADD_FUNCTION
			optim: ET_ADAM
			params: ARRAYED_LIST [ET_TENSOR]
			step: INTEGER_32
			loss, pred_val, target_val: REAL_64
			grad_pred: ET_TENSOR
		do
			print ("%N[1.6] Tiny Training Loop (y = w*x + b)%N")
			
			-- Prepare data: x = 2.0, target = 10.0
			x := {ET_TORCH}.tensor (<<1>>)
			if attached {ET_STORAGE_REAL_64} x.storage as s then s.put_real_64 (2.0, 1) end
			
			target := {ET_TORCH}.tensor (<<1>>)
			if attached {ET_STORAGE_REAL_64} target.storage as s then s.put_real_64 (10.0, 1) end
			
			-- Prepare parameters: w = 1.0, b = 1.0
			w := {ET_TORCH}.ones (<<1>>)
			w.set_requires_grad (True)
			b := {ET_TORCH}.ones (<<1>>)
			b.set_requires_grad (True)
			
			-- Optimizer
			create params.make (2)
			params.extend (w)
			params.extend (b)
			create optim.make (params, 0.1) -- lr=0.1
			
			create l_mul
			create l_add
			
			from step := 1 until step > 50 loop
				-- 1. Forward pass
				create v_w.make (w)
				create v_x.make (x)
				create v_b.make (b)
				
				v_mul := l_mul.forward (<<v_w, v_x>>)
				v_pred := l_add.forward (<<v_mul, v_b>>)
				
				if attached {ET_STORAGE_REAL_64} v_pred.data.storage as spred and attached {ET_STORAGE_REAL_64} target.storage as starget then
					pred_val := spred.item_as_real_64 (1)
					target_val := starget.item_as_real_64 (1)
					
					-- Loss = 0.5 * (pred - target)^2
					loss := 0.5 * (pred_val - target_val) * (pred_val - target_val)
					
					if step \\ 10 = 0 or step = 1 then
						print ("Step " + step.out + " | Loss: " + loss.out + " | Pred: " + pred_val.out + "%N")
					end
					
					-- 2. Backward pass
					-- dL/d(pred) = (pred - target)
					grad_pred := {ET_TORCH}.tensor (<<1>>)
					if attached {ET_STORAGE_REAL_64} grad_pred.storage as gs then
						gs.put_real_64 (pred_val - target_val, 1)
					end
					v_pred.set_grad (grad_pred)
					v_pred.backward
					
					-- 3. Optimize
					optim.step
					optim.zero_grad
				end
				
				step := step + 1
			end
			
			print ("Training Finished.%N")
			if attached {ET_STORAGE_REAL_64} w.storage as ws and attached {ET_STORAGE_REAL_64} b.storage as bs then
				print ("Optimized Weight w: " + ws.item_as_real_64 (1).out + "%N")
				print ("Optimized Bias b: " + bs.item_as_real_64 (1).out + "%N")
			end
		end

	test_math
		local
			d: REAL_64
			m: DOUBLE_MATH
		do
			print ("%N[1.7] Testing DOUBLE_MATH%N")
			create m
			d := m.sqrt (4.0)
			print ("Sqrt of 4.0: " + d.out + "%N")
			d := m.exp (1.0)
			print ("Exp of 1.0: " + d.out + "%N")
		end

	test_delivery_prediction
			-- The Machine Learning Pipeline Lab (Delivery Scenario)
		local
			distances, times: ET_TENSOR
			d_arr, t_arr, n_arr: ARRAY [ARRAY [REAL_64]]
			lin: ET_LINEAR
			mod_array: ARRAY [ET_MODULE]
			model: ET_SEQUENTIAL
			loss_fn: ET_MSE_LOSS
			optim: ET_SGD
			epoch: INTEGER_32
			outputs, loss: ET_TENSOR
			no_grad_agent: PROCEDURE
			l_layer: ET_MODULE
		do
			print ("%N[1.8] Delivery Prediction Lab (Notebook translation)%N")
			
			d_arr := << << 1.0 >>, << 2.0 >>, << 3.0 >>, << 4.0 >> >>
			create distances.make_from_array2d (d_arr)
			
			t_arr := << << 6.96 >>, << 12.11 >>, << 16.77 >>, << 22.21 >> >>
			create times.make_from_array2d (t_arr)

			create mod_array.make_empty
			create lin.make (1, 1)
			mod_array.force (lin, 1)
			
			create model.make (mod_array)
			create loss_fn
			create optim.make (model.parameters, 0.01)

			from epoch := 1 until epoch > 500 loop
				optim.zero_grad
				outputs := model.forward (distances)
				loss := loss_fn.forward (outputs, times)
				loss.backward
				
				if epoch = 1 then
					print ("--- Epoch 1 Debug ---%N")
					print ("Outputs: " + outputs.out + "%N")
					print ("Loss: " + loss.scalar_value.out + "%N")
					l_layer := model[1]
					if attached {ET_LINEAR} l_layer as l_lin then
						if attached l_lin.weight.grad as wg then
							print ("Weight grad: " + wg.out + "%N")
						else
							print ("Weight grad: Void%N")
						end
						if attached l_lin.bias as lb and then attached lb.grad as bg then
							print ("Bias grad: " + bg.out + "%N")
						else
							print ("Bias grad: Void%N")
						end
					end
					print ("--- End Debug ---%N")
				end
				
				optim.step
				
				if epoch \\ 50 = 0 then
					print ("Epoch " + epoch.out + ": Loss = " + loss.scalar_value.out + "%N")
				end
				epoch := epoch + 1
			end

			l_layer := model[1]
			if attached {ET_LINEAR} l_layer as l_lin then
				print ("Weight: " + l_lin.weight.scalar_value.out + "%N")
				if attached l_lin.bias as l_bias then
					print ("Bias: " + l_bias.scalar_value.out + "%N")
				end
			end

			no_grad_agent := agent (seq_model: ET_SEQUENTIAL)
				local
					l_n_arr: ARRAY [ARRAY [REAL_64]]
					l_new_distance, l_predicted_time, l_matmul_result: ET_TENSOR
					l_debug_layer: ET_MODULE
					l_w_t: ET_TENSOR
				do
					l_n_arr := << << 7.0 >> >>
					create l_new_distance.make_from_array2d (l_n_arr)
					
					-- Debug: manual forward to see intermediate results
					l_debug_layer := seq_model[1]
					if attached {ET_LINEAR} l_debug_layer as dl then
						print ("--- Prediction Debug ---%N")
						print ("Input: " + l_new_distance.out + "%N")
						print ("Weight storage val: " + dl.weight.storage.item_as_real_64 (1).out + "%N")
						l_w_t := dl.weight.transpose_internal (1, 2)
						print ("Weight^T storage val: " + l_w_t.storage.item_as_real_64 (1).out + "%N")
						l_matmul_result := l_new_distance.matmul_internal (l_w_t)
						print ("matmul_internal result: " + l_matmul_result.scalar_value.out + "%N")
						if attached dl.bias as db then
							print ("Bias val: " + db.scalar_value.out + "%N")
						end
						print ("--- End Prediction Debug ---%N")
					end
					
					l_predicted_time := seq_model.forward (l_new_distance)
					print ("Prediction for a 7.0-mile delivery: " + l_predicted_time.scalar_value.out + " minutes%N")
					
					if l_predicted_time.scalar_value > 30.0 then
						print ("Decision: Do NOT take the job. You will likely be late.%N")
					else
						print ("Decision: Take the job. You can make it!%N")
					end
				end (model)
			{ET_TORCH}.with_no_grad (no_grad_agent)
		end

end
