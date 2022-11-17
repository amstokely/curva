
%inline %{
class StopNodeIterator {};
class NodeIterator {
		public:
		NodeIterator(
				ForwardIterator<Atom*> _cur,
				ForwardIterator<Atom*> _end
		) : cur(_cur), end(_end) {}
		NodeIterator* __iter__()
		{
			return this;
		}
		ForwardIterator<Atom*> cur;
		ForwardIterator<Atom*> end;
};
%}
%include "exception.i"
%exception NodeIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (StopNodeIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend NodeIterator {
		Atom& __next__() {
			if ($self->cur != $self->end){
				return *(*$self->cur++);
			}
			throw StopNodeIterator();
		}
}

%extend Node {
		NodeIterator __iter__() {
			return NodeIterator($self->begin(), $self->end());
		}
};
%inline %{
class StopAtomsIterator {};
class AtomsIterator {
		public:
		AtomsIterator(
				RandomAccessIterator<Atom> _cur,
				RandomAccessIterator<Atom> _end
		) : cur(_cur), end(_end) {}
		AtomsIterator* __iter__()
		{
			return this;
		}
		RandomAccessIterator<Atom> cur;
		RandomAccessIterator<Atom> end;
};
%}
%include "exception.i"
%exception AtomsIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (StopAtomsIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend AtomsIterator {
		Atom& __next__() {
			if ($self->cur != $self->end){
				return *$self->cur++;
			}
			throw StopAtomsIterator();
		}
}

%extend Atoms {
		AtomsIterator __iter__() {
			return AtomsIterator($self->begin(), $self->end());
		}
};

%inline %{
class StopNodesIterator {};
class NodesIterator {
		public:
		NodesIterator(
				MapValueForwardIterator<std::string, Node*> _cur,
				MapValueForwardIterator<std::string, Node*> _end
		) : cur(_cur), end(_end) {}
		NodesIterator* __iter__()
		{
			return this;
		}
		MapValueForwardIterator<std::string, Node*> cur;
		MapValueForwardIterator<std::string, Node*> end;
};
%}
%include "exception.i"
%exception NodesIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (StopNodesIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend NodesIterator {
		Node* __next__() {
			if ($self->cur != $self->end){
				return *$self->cur++;
			}
			throw StopNodesIterator();
		}
}

%extend Nodes {
		NodesIterator __iter__() {
			return NodesIterator($self->begin(), $self->end());
		}
};

%inline %{
class dStopCurvaVectorIterator {};
class dCurvaVectorIterator {
		public:
		dCurvaVectorIterator(
				RandomAccessIterator<double> _cur,
				RandomAccessIterator<double> _end
		) : cur(_cur), end(_end) {}
		dCurvaVectorIterator* __iter__()
		{
			return this;
		}
		RandomAccessIterator<double> cur;
		RandomAccessIterator<double> end;
};
%}
%include "exception.i"
%exception dCurvaVectorIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (dStopCurvaVectorIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend dCurvaVectorIterator {
		double __next__() {
			if ($self->cur != $self->end){
				return *$self->cur++;
			}
			throw dStopCurvaVectorIterator();
		}
}

%extend CurvaVector<double> {
		dCurvaVectorIterator __iter__() {
			return dCurvaVectorIterator($self->begin(), $self->end());
		}
};

%inline %{
class iStopCurvaVectorIterator {};
class iCurvaVectorIterator {
		public:
		iCurvaVectorIterator(
				RandomAccessIterator<int> _cur,
		RandomAccessIterator<int> _end
		) : cur(_cur), end(_end) {}
		iCurvaVectorIterator* __iter__()
		{
			return this;
		}
		RandomAccessIterator<int> cur;
		RandomAccessIterator<int> end;
};
%}
%include "exception.i"
%exception iCurvaVectorIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (iStopCurvaVectorIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend iCurvaVectorIterator {
		int __next__() {
			if ($self->cur != $self->end){
				return *$self->cur++;
			}
			throw iStopCurvaVectorIterator();
		}
}

%extend CurvaVector<int> {
		iCurvaVectorIterator __iter__() {
			return iCurvaVectorIterator($self->begin(), $self->end());
		}
};

%inline %{
class dStopCurvaMatrixIterator {};
class dCurvaMatrixIterator {
		public:
		dCurvaMatrixIterator(
			std::vector<int>::iterator _cur,
			std::vector<int>::iterator _end,
			CurvaMatrix<double> *curvaMatrix_
		) : cur(_cur), end(_end), curvaMatrix(curvaMatrix_){}
		dCurvaMatrixIterator* __iter__()
		{
			return this;
		}
		std::vector<int>::iterator cur;
		std::vector<int>::iterator end;
		CurvaMatrix<double> *curvaMatrix;
};
%}
%include "exception.i"
%exception dCurvaMatrixIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (dStopCurvaMatrixIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend dCurvaMatrixIterator {
		CurvaVector<double> __next__() {
			if ($self->cur != $self->end){
				CurvaVector<double> curvaVector;
				unsigned int rowIndex = *$self->cur;
				$self->cur++;
				curvaVector.init(
						$self->curvaMatrix, rowIndex
						);
				return curvaVector;
			}
			throw dStopCurvaMatrixIterator();
		}
}

%extend CurvaMatrix<double> {
		dCurvaMatrixIterator __iter__() {
			return dCurvaMatrixIterator($self->rowIndicesBegin(),
									   $self->rowIndicesEnd(),
									   $self
									   );
		}
};

%inline %{
class iStopCurvaMatrixIterator {};
class iCurvaMatrixIterator {
		public:
		iCurvaMatrixIterator(
				std::vector<int>::iterator _cur,
		std::vector<int>::iterator _end,
		CurvaMatrix<int> *curvaMatrix_
		) : cur(_cur), end(_end), curvaMatrix(curvaMatrix_){}
		iCurvaMatrixIterator* __iter__()
		{
			return this;
		}
		std::vector<int>::iterator cur;
		std::vector<int>::iterator end;
		CurvaMatrix<int> *curvaMatrix;
};
%}
%include "exception.i"
%exception iCurvaMatrixIterator::__next__ {
		try {
			$action // calls %extend function next() below
		}
		catch (iStopCurvaMatrixIterator) {
			PyErr_SetString(PyExc_StopIteration, "End of iterator");
			return NULL;
		}
}
%extend iCurvaMatrixIterator {
		CurvaVector<int> __next__() {
			if ($self->cur != $self->end){
				CurvaVector<int> curvaVector;
				unsigned int rowIndex = *$self->cur;
				$self->cur++;
				curvaVector.init(
						$self->curvaMatrix, rowIndex
				);
				return curvaVector;
			}
			throw iStopCurvaMatrixIterator();
		}
}

%extend CurvaMatrix<int> {
		iCurvaMatrixIterator __iter__() {
			return iCurvaMatrixIterator($self->rowIndicesBegin(),
			                                 $self->rowIndicesEnd(),
			                                 $self
			);
		}
};
