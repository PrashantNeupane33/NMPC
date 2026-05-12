CXX      = g++
CXXFLAGS = -I/usr/include/eigen3 -I/usr/local/include -Iinclude
LDFLAGS  = -L/usr/local/lib -lcasadi -latomic
MAKEFLAGS += -j$(nproc)

TARGET  = simulation.out
SRCS    = src/driver.cpp src/Controller.cpp src/Observer.cpp src/Pathfinder.cpp
HEADERS = include/Controller.hpp include/Observer.hpp

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(SRCS) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
