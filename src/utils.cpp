//
// Created by andy on 7/24/22.
//
#include "../include/utils.h"
#include <numeric>
#include  <algorithm>

void utils::setLastFrame (
		int *lastFrame,
		int numFrames
) {
	if (*lastFrame == -1) {
		*lastFrame = numFrames - 1;
	}
}

void utils::generateIndicesArray (
		unsigned int **indicesArray,
		int size
) {
	*indicesArray = new unsigned int[size];
	std::iota(*indicesArray, *indicesArray + size, 0);
}

bool utils::isRecordAtom (std::string &pdbLine) {
	size_t isRecordAtom = pdbLine.find("ATOM");
	if (isRecordAtom != std::string::npos) {
		return true;
	} else {
		return false;
	}
}

bool utils::isEndOfFrame (std::string &pdbLine) {
	size_t isEndOfFrame = pdbLine.find("END");
	if (isEndOfFrame != std::string::npos) {
		return true;
	} else {
		return false;
	}
}

std::string utils::removeWhiteSpace (std::string str) {
	str.erase(
			std::remove_if(str.begin(), str.end(), isspace),
			str.end()
	);
	return str;
}

double utils::strToDouble (const std::string &str) {
	return std::stod(str);
}

int utils::strToInt (const std::string &str) {
	return std::stoi(str);
}

unsigned int utils::hashString (
		const std::string &str
) {
	const char   *strPtr = str.data();
	unsigned int h       = HASH_FIRSTH;
	while (*strPtr) {
		h = (h * HASH_A) ^ (strPtr[0] * HASH_B);
		strPtr++;
	}
	return h;
}

