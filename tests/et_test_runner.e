note
	description: "Dummy Root class for Eiffel AutoTest execution."

class
	ET_TEST_RUNNER

create
	make

feature -- Test classes

	test_classes: ARRAY [TYPE [EQA_TEST_SET]]
		once
			Result := <<
				{ET_TEST_UTILS},
				{ET_TEST_AUTOGRAD},
				{ET_TEST_TENSOR},
				{ET_TEST_KV_CACHE},
				{ET_TEST_SAFE_TENSORS}
			>>
		end

feature -- Initialization

	make
		do
			print ("This is a dummy root. Run this target using ec -tests.%N")
		end

end
