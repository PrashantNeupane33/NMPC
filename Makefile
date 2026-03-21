CXX      = g++
CXXFLAGS = -I/usr/include/eigen3 -I/usr/local/include
LDFLAGS  = -L/usr/local/lib -lcasadi
MAKEFLAGS += -j$(nproc)

TARGET  = simulation.out
SRCS    = driver.cpp Controller.cpp Observer.cpp
HEADERS = Bezier.hpp Controller.hpp Observer.hpp

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(SRCS) $(HEADERS) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
