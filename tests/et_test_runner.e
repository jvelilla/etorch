note
	description: "Dummy Root class for Eiffel AutoTest execution."

class
	ET_TEST_RUNNER

create
	make

feature -- Initialization

	make
		do
			print ("This is a dummy root. Run this target using ec -tests.%N")
		end

end
