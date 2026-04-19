CXX=mpicxx
CXXFLAGS=-O2 -std=c++17

TARGETS=q1 q2 q3 q4 q5

all: $(TARGETS)

q1: q1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q2: q2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q3: q3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q4: q4.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

q5: q5.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
