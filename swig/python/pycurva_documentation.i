/*
 * 	void init (
			const std::string &dcd,
			const std::string &pdb,
			int firstFrame,
			int lastFrame,
			int windowSize,
			const std::string &name
	);
 */

%feature("autodoc", "3");
%typemap("doc") const std::string& dcd "$1_name (C++ type: $1_type) -- "
								  "Input "
						  "$1_name "
					"dimension"
void MolecularDynamicsCalculation::init (
	const std::string &dcd,
	const std::string &pdb,
	int firstFrame,
	int lastFrame,
	int windowSize,
	const std::string &name
);
