DTCXXFLAGS = -std=c++11 -g -I$(DAQROOT)/include -I..

DTCXXLDFLAGS=-L$(DAQLIB) -ldataformat -ldaqio \
	-lDataFlow -lException -lgsl -lgslcblas	\
	-Wl,-rpath=$(DAQLIB)


all: DtRange

DtRange: DtRange.o
	$(CXX) $(DTCXXLDFLAGS) -o DtRange DtRange.o ../functions.o ../lmfit.o

DtRange.o: DtRange.cpp
	$(CXX) $(DTCXXFLAGS) -c DtRange.cpp

clean:
	rm -f DtRange *.o

INSTDIR=/usr/opt/spectcl/5.0-005
