DTCXXFLAGS = -std=c++11 -g -I$(DAQROOT)/include -I..

DTCXXLDFLAGS=-L$(DAQLIB) -ldataformat -ldaqio \
	-lDataFlow -lException -lgsl -lgslcblas	\
	-Wl,-rpath=$(DAQLIB)


all: DtRange SpecTcl

DtRange: DtRange.o
	$(CXX) $(DTCXXLDFLAGS) -o DtRange DtRange.o ../functions.o

DtRange.o: DtRange.cpp
	$(CXX) $(DTCXXFLAGS) -c DtRange.cpp

INSTDIR=/usr/opt/spectcl/5.0-005
# Skeleton makefile for 3.1

include $(INSTDIR)/etc/SpecTcl_Makefile.include

#  If you have any switches that need to be added to the default c++ compilation
# rules, add them to the definition below:

USERCXXFLAGS=-I. -I.. -std=c++11 -g

#  If you have any switches you need to add to the default c compilation rules,
#  add them to the defintion below:

USERCCFLAGS=$(USERCXXFLAGS)

#  If you have any switches you need to add to the link add them below:

USERLDFLAGS=-lgsl -lgslcblas

#
#   Append your objects to the definitions below:
#

OBJECTS=MySpecTclApp.o ../functions.o REventProcessor.o

#
#  Finally the makefile targets.
#


SpecTcl: $(OBJECTS)
	$(CXXLD)  -o SpecTcl $(OBJECTS) $(USERLDFLAGS) \
	$(LDFLAGS)


REventProcessor.o: REventProcessor.cpp REventProcessor.h ../functions.h
	$(CXX) -c $(CXXFLAGS) $(USERCXXFLAGS) REventProcessor.cpp


clean:
	rm -f *.o SpecTcl DtRange

depend:
	makedepend $(USERCXXFLAGS) *.cpp *.c

help:
	echo "make                 - Build customized SpecTcl"
	echo "make clean           - Remove objects from previous builds"
	echo "make depend          - Add dependencies to the Makefile. "
