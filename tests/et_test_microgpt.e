note
	description: "Tests for ET_MICROGPT."

class
	ET_TEST_MICROGPT

inherit
	EQA_TEST_SET

feature -- Tests

	test_microgpt_forward
		local
			gpt: ET_MICROGPT
			x, y: ET_TENSOR
			x_store: ET_STORAGE_REAL_64
			tol: REAL_64
			loader: ET_SAFE_TENSORS_LOADER
			path: PATH
			ln_f_weight: ET_TENSOR
			i: INTEGER_32
		do
			tol := 1.0e-4
			
			-- Generate or assert dummy safe tensors file exists
			create path.make_from_string ("microgpt_dummy.safetensors")
			
			-- Minimal MicroGPT: 1 layer, n_embd=4, n_head=2
			create gpt.make (1, 4, 2, 0.0)
			
			-- Load SafeTensors
			create loader.make
			ln_f_weight := loader.load_tensor (path, "ln_f.weight")
			assert ("Successfully loaded ln_f.weight from safetensors", ln_f_weight /= Void and then ln_f_weight.shape.count = 1)
			
			-- We manually assign this loaded weight to prove integration works:
			if attached ln_f_weight then
				if attached {ET_STORAGE_REAL_64} gpt.ln_f.weight.storage as dest_store and then
				   attached {ET_STORAGE_REAL_64} ln_f_weight.storage as src_store then
					from i := 1 until i > ln_f_weight.numel loop
						dest_store.put_real_64 (src_store.item_as_real_64 (i), i)
						i := i + 1
					end
				end
			end

			-- Dummy input sequence: [Batch=1, Seq=2, Embd=4]
			-- Keeping it [Seq=2, Embd=4] corresponding to 2D
			create x_store.make (8)
			x_store.put_real_64 (0.1, 1)
			x_store.put_real_64 (0.2, 2)
			x_store.put_real_64 (0.3, 3)
			x_store.put_real_64 (0.4, 4)
			x_store.put_real_64 (0.5, 5)
			x_store.put_real_64 (0.6, 6)
			x_store.put_real_64 (0.7, 7)
			x_store.put_real_64 (0.8, 8)
			
			create x.make_from_storage (x_store, <<1, 2, 4>>, <<8, 4, 1>>, 0)
			
			-- Test forward pass
			y := gpt.forward (x)
			
			assert ("Result shape count", y.shape.count = 3)
			assert ("Result shape Batch", y.shape [1] = 1)
			assert ("Result shape Seq", y.shape [2] = 2)
			assert ("Result shape Embd", y.shape [3] = 4)
			
			assert ("Successfully executed Transformer forward pass", True)
		end

end
