//
// Created by andy on 7/24/22.
//

#ifndef CURVA_UTILS_H
#define CURVA_UTILS_H
#define HASH_A 54059 /* a prime */
#define HASH_B 76963 /* another prime */
#define HASH_FIRSTH 37 /* also prime */


#include <string>

namespace utils {
	void setLastFrame (
			int *lastFrame,
			int numFrames
	);

	void generateIndicesArray (
			unsigned int **indicesArray,
			int size
	);

	bool isRecordAtom (std::string &pdbLine);

	bool isEndOfFrame (std::string &pdbLine);

	double strToDouble (const std::string &str);

	int strToInt (const std::string &str);

	std::string removeWhiteSpace (std::string str);

	unsigned int hashString(const std::string& str);

}
#endif //CURVA_UTILS_H
