CXXFLAGS = -std=c++11 -g -I$(DAQROOT)/include -I..

CXXLDFLAGS=-L$(DAQLIB) -ldataformat -ldaqio \
	-lDataFlow -lException -lgsl -lgslcblas	\
	-Wl,-rpath=$(DAQLIB)


all: DtRange

DtRange: DtRange.o
	$(CXX) $(CXXLDFLAGS) -o DtRange DtRange.o ../functions.o

DtRange.o: DtRange.cpp
	$(CXX) $(CXXFLAGS) -c DtRange.cpp
