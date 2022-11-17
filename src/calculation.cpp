//
// Created by andy on 8/18/22.
//

#include "../include/calculation.h"
#include "../include/curva.h"
#include "../include/cnpy/cnpy.h"
#include "../include/cuda/mutual_information.h"
#include "../include/cuda/pearson_correlation.h"
#include "../include/cuda/cutoff_matrix.h"

#define BOOST_NO_CXX11_SCOPED_ENUMS

#include <boost/filesystem.hpp>

#undef BOOST_NO_CXX11_SCOPED_ENUMS


MolecularDynamicsCalculation::MolecularDynamicsCalculation () {
	this->_atoms                    = new Atoms;
	this->_nodes                    = new Nodes;
	this->XY_                       = new CurvaMatrix<double>;
	this->_mutualInformationMatrix  = new CurvaMatrix<double>;
	this->_pearsonCorrelationMatrix = new CurvaMatrix<double>;
}

MolecularDynamicsCalculation::~MolecularDynamicsCalculation () {
	delete _atoms;
	delete _nodes;
	delete XY_;
	delete _mutualInformationMatrix;
	delete _pearsonCorrelationMatrix;
}

void MolecularDynamicsCalculation::init (
		const std::string &dcd,
		const std::string &pdb,
		int firstFrame,
		int lastFrame,
		int windowSize,
		const std::string &name
) {
	auto *coordinates = new Coordinates;
	this->_atoms                    = new Atoms;
	this->_nodes                    = new Nodes;
	this->XY_                       = new CurvaMatrix<double>;
	this->_mutualInformationMatrix  = new CurvaMatrix<double>;
	this->_pearsonCorrelationMatrix = new CurvaMatrix<double>;
	this->_pdb                      = pdb;
	this->_dcd                      = dcd;
	this->_numAtoms                 = 0;
	this->_numFrames                = 0;
	this->_windowSize               = windowSize;
	this->_name                     = name;
	curva::parseDcd(
			this->_dcd,
			coordinates,
			&this->_numAtoms,
			&this->_numFrames,
			firstFrame,
			lastFrame
	);
	curva::parsePdb(
			this->_pdb,
			this->_atoms
	);
	curva::generateNodes(
			this->_nodes,
			this->_atoms,
			coordinates,
			this->_numFrames / this->_windowSize
	);
	this->_numNodes = (int) this->_nodes->numNodes();
	delete coordinates;
}

Atoms *MolecularDynamicsCalculation::atoms () {
	return this->_atoms;
}

Nodes *MolecularDynamicsCalculation::nodes () {
	return this->_nodes;
}

int MolecularDynamicsCalculation::numFrames () const {
	return this->_numFrames;
}

int MolecularDynamicsCalculation::numAtoms () const {
	return this->_numAtoms;
}

int MolecularDynamicsCalculation::numNodes () const {
	return this->_numNodes;
}

std::string MolecularDynamicsCalculation::dcd () {
	return this->_dcd;
}

std::string MolecularDynamicsCalculation::pdb () {
	return this->_pdb;
}

void MolecularDynamicsCalculation::mutualInformationAllocate (
		CurvaMatrix<double> *X
) {
	X->init(
			3 * this->numFrames() * this->numNodes(), 1
	);
	for (unsigned int I = 0;
	     I < this->numNodes();
	     I++) {
		for (unsigned int i = 0;
		     i < this->numFrames();
		     i++) {
			unsigned int offset = I * (3 * this->numFrames());
			(*(X))(offset + i, 0)
					=
					this->nodes()->at(I)->centerOfMass().at(3 * i);
			(*(X))(
					offset + i + this->numFrames(),
					0
			) =
					this->nodes()->at(I)->centerOfMass().at(3 * i + 1);
			(*(X))(
					offset + i + 2 * this->numFrames(), 0
			) =
					this->nodes()->at(I)->centerOfMass().at(3 * i + 2);
		}
	}
	X->allocate();
	cudaNormalizeNodeCoordinates(
			X,
			this->_windowSize,
			this->numNodes(), this->_numFrames
	);
}

void MolecularDynamicsCalculation::mutualInformation (
		int referenceIndex,
		double cutoff,
		int k,
		const std::string &norm
) {
	if (!this->XY_->host()) {
		this->mutualInformationAllocate(
				this->XY_
		);
	}
	if (!this->_mutualInformationMatrix->host()) {
		this->_mutualInformationMatrix->init(
				this->numNodes(),
				this->numNodes()
		);
	}
	auto *averageNodePositionMatrix = new CurvaMatrix<double>;
	averageNodePositionMatrix->init(
			3, this->_numNodes
	);
	averageNodePositionMatrix->allocate();
	auto *cutoffMatrix = new CurvaMatrix<double>;
	cutoffMatrix->init(
			this->numNodes(),
			this->numNodes()
	);
	cutoffMatrix->allocate();
	for (unsigned int windowIndex = 0;
	     windowIndex < this->_numFrames / this->_windowSize;
	     windowIndex++) {
		auto nodesIterator = this->nodes()->begin();
		while (nodesIterator != this->nodes()->end()) {
			curva::mutualInformationConstruct(
					*nodesIterator,
					averageNodePositionMatrix, windowIndex,
					this->_windowSize
			);
			nodesIterator++;
		}
		cudaCutoffMatrix(
				averageNodePositionMatrix,
				cutoffMatrix,
				cutoff,
				this->_numNodes,
				this->_windowSize
		);
		cudaMutualInformation(
				this->XY_,
				this->_mutualInformationMatrix, cutoffMatrix,
				this->numNodes(),
				this->_windowSize, this->_numFrames, windowIndex,
				referenceIndex, k, norm
		);
	}
	for (unsigned int i           = 0;
	     i < this->_numNodes;
	     i++) {
		(*(this->_mutualInformationMatrix))(referenceIndex, i)
				/= ((double) this->_numFrames / this->_windowSize);
	}
	delete averageNodePositionMatrix;
	delete cutoffMatrix;
}

void MolecularDynamicsCalculation::pearsonCorrelation (double cutoff) {
	auto *deltaAveragePositionMatrix =
			     new CurvaMatrix<double>;
	deltaAveragePositionMatrix->init(
			this->nodes()->numNodes(),
			this->numFrames()
	);
	auto *averageNodePositionMatrix =
			     new CurvaMatrix<double>;
	averageNodePositionMatrix->init(
			3, this->nodes()->numNodes()
	);
	auto *cutoffMatrix = new CurvaMatrix<double>;
	cutoffMatrix->init(
			this->nodes()->numNodes(),
			this->nodes()->numNodes()
	);
	this->_pearsonCorrelationMatrix->init(
			this->nodes()->numNodes(),
			this->nodes()->numNodes()
	);
	deltaAveragePositionMatrix->allocate();
	this->_pearsonCorrelationMatrix->allocate();
	averageNodePositionMatrix->allocate();
	cutoffMatrix->allocate();
	for (unsigned int windowIndex = 0;
	     windowIndex < this->_numFrames / this->_windowSize;
	     windowIndex++) {


		auto nodesIterator = this->nodes()->begin();
		while (nodesIterator != this->nodes()->end()) {
			curva::pearsonCorrelationConstruct(
					*nodesIterator,
					deltaAveragePositionMatrix,
					averageNodePositionMatrix, windowIndex,
					this->_windowSize
			);
			nodesIterator++;
		}
		cudaCutoffMatrix(
				averageNodePositionMatrix,
				cutoffMatrix,
				cutoff,
				this->nodes()->numNodes(),
				this->_windowSize
		);
		cudaPearsonCorrelation(
				deltaAveragePositionMatrix,
				this->_pearsonCorrelationMatrix,
				cutoffMatrix,
				this->nodes()->numNodes(),
				this->_windowSize
		);
	}
	for (auto         &val: *this->_pearsonCorrelationMatrix) {
		val /= ((double) this->_numFrames / this->_windowSize);
	}
	delete deltaAveragePositionMatrix;
	delete averageNodePositionMatrix;
	delete cutoffMatrix;
}

CurvaMatrix<double> *
MolecularDynamicsCalculation::mutualInformationMatrix () const {
	return this->_mutualInformationMatrix;
}

CurvaMatrix<double> *
MolecularDynamicsCalculation::pearsonCorrelationMatrix () const {
	return this->_pearsonCorrelationMatrix;
}

void MolecularDynamicsCalculation::save (
		const std::string &fname,
		int indent
) {
	std::string    serializationDirectory = boost::filesystem::path(
			boost::filesystem::absolute(fname)
	).parent_path().string();
	nlohmann::json j{
			{"_nodes",      this->_nodes->save(
					serializationDirectory, this->_name
			)},
			{"_dcd",        this->_dcd},
			{"_pdb",        this->_pdb},
			{"_numAtoms",   this->_numAtoms},
			{"_numFrames",  this->_numFrames},
			{"_windowSize", this->_windowSize},
			{"_numNodes",   this->_numNodes},
			{"_name",       this->_name}
	};
	std::ofstream  o;
	o.open(fname);
	o << j.dump(indent);
	o.close();
}

void MolecularDynamicsCalculation::load (const std::string &fname) {
	this->_atoms = new Atoms;
	nlohmann::json j;
	std::ifstream  i(fname);
	i >> j;
	i.close();
	std::string serializationDirectory = boost::filesystem::path(
			boost::filesystem::absolute(fname)
	).parent_path().string();
	this->_nodes = new Nodes;
	this->_nodes->load(
			j.at("_nodes").get<nlohmann::json>(), serializationDirectory
	);
	this->_dcd        = j.at("_dcd").get<std::string>();
	this->_pdb        = j.at("_pdb").get<std::string>();
	this->_numAtoms   = j.at("_numAtoms").get<int>();
	this->_numFrames  = j.at("_numFrames").get<int>();
	this->_windowSize = j.at("_windowSize").get<int>();
	this->_numNodes   = j.at("_numNodes").get<int>();
	this->_name       = j.at("_name").get<std::string>();
	for (auto node: *(this->_nodes)) {
		for (auto atom: *node) {
			this->_atoms->addAtom(*atom);
		}
	}
	std::sort(
			this->_atoms->begin(),
			this->_atoms->end(),
			[] (
					const Atom &a,
					const Atom &b
			) -> bool {
				return (
						a.index() < b.index()
				);
			}
	);
}


