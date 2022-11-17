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

	auto *generalizedCorrelationMatrix = new CurvaMatrix<double>;
	curva::test::generalizedCorrelationTest(
			generalizedCorrelationMatrix,
			xfname,
			yfname, 0
	);
	for (auto val: *generalizedCorrelationMatrix) {
		std::cout << val << std::endl;
	}
	delete generalizedCorrelationMatrix;
	return 0;
}