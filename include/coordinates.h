//
// Created by andy on 7/21/22.
//

#ifndef CURVA_COORDINATES_H
#define CURVA_COORDINATES_H

#include "custom_iterators/random_access_iterator.h"
#include <vector>

class Coordinates {
public:
	Coordinates ();

	~Coordinates ();

	void init (
			unsigned int numAtoms_, unsigned int numFrames_);

	double x (int index, int frame);

	double y (int index, int frame);

	double z (int index, int frame);

	double &operator() (int i,
	                    int j, int k);

	unsigned int numAtoms() const;

	unsigned int numFrames() const;

	RandomAccessIterator<double> begin () {
		return RandomAccessIterator<double>(&coordinates[0]);
	}

	RandomAccessIterator<double> end () {
		return RandomAccessIterator<double>
		        (&coordinates[3*_numAtoms*_numFrames]);
	}

private:
	std::vector<double> coordinates;
	unsigned int _numAtoms;
	unsigned int _numFrames;
};

#endif //CURVA_COORDINATES_H
