VPATH=..
CXXFLAGS=-I.. -std=c++11 -g


pulsemaker: pulsemaker.o functions.o
	$(CXX) $(CXXFLAGS) -o $@ $^  -lm

clean:
	rm -f *.o pulsemaker
