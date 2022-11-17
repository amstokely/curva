//
// Created by andy on 7/22/22.
//
#include <random>
#include "../include/curva.h"
#include "../include/matrix/curva_vector.h"
#include "../include/cnpy/cnpy.h"

int main () {
	std::string yfname = "./y.npy";
	std::string xfname = "./x.npy";

	auto *mutualInformationMatrix = new CurvaMatrix<double>;
	curva::test::mutualInformationTest(
			mutualInformationMatrix,
			xfname,
			yfname, 0
	);
	for (auto val: *mutualInformationMatrix) {
		std::cout << val << std::endl;
	}
	delete mutualInformationMatrix;
	return 0;
}