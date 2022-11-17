
%module tclcurva

%include std_string.i
%include std_vector.i
%template(dVector) std::vector<double>;
%template(ddVector) std::vector<std::vector<double>>;
%template(iVector) std::vector<int>;

%include "tclcurva_headers1.i"
%include "tclcurva_rename.i"

%template(nodeVector) std::vector<Node>;
%template(atomVector) std::vector<Atom>;
%template(curvaVectorDoubleVector) std::vector<CurvaVector<double>>;
%template(curvaVectorIntVector) std::vector<CurvaVector<int>>;

%include "tclcurva_ignore.i"
%include "tclcurva_typemaps.i"
%include "tclcurva_headers2.i"
%template(dCurvaVector) CurvaVector<double>;
%template(iCurvaVector) CurvaVector<int>;
%template(dCurvaMatrix) CurvaMatrix<double>;
%template(iCurvaMatrix) CurvaMatrix<int>;
