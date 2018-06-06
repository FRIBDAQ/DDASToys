INSTDIR=/usr/opt/spectcl/experimental/5.0-007
# Skeleton makefile for 3.1

include $(INSTDIR)/etc/SpecTcl_Makefile.include

#  If you have any switches that need to be added to the default c++ compilation
# rules, add them to the definition below:

USERCXXFLAGS=-I.. -I$(DAQROOT)/include -I$(DDAS_INC)

#  If you have any switches you need to add to the default c compilation rules,
#  add them to the defintion below:

USERCCFLAGS=$(USERCXXFLAGS)

#  If you have any switches you need to add to the link add them below:

USERLDFLAGS=../FitHitUnpacker.o -L$(DAQLIB) -lFragmentIndex -ldataformat   -Wl,"-rpath=$(DAQLIB)" \
	-L$(DDAS_LIB) -lddasformat -Wl,"-rpath=$(DDAS_LIB)"

#
#   Append your objects to the definitions below:
#

OBJECTS=MySpecTclApp.o FitEventProcessor.o

#
#  Finally the makefile targets.
#


SpecTcl: $(OBJECTS)
	$(CXXLD)  -o SpecTcl $(OBJECTS) $(USERLDFLAGS) \
	$(LDFLAGS) 


clean:
	rm -f $(OBJECTS) SpecTcl

depend:
	makedepend $(USERCXXFLAGS) *.cpp *.c

help:
	echo "make                 - Build customized SpecTcl"
	echo "make clean           - Remove objects from previous builds"
	echo "make depend          - Add dependencies to the Makefile. "
