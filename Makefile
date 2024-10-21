# Variables
CXX = g++
CXXFLAGS = -Wall -std=c++11
TARGET = main
OBJS = main.o functions.o

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS)

# Compiling main.cpp
main.o: main.cpp functions.h
	$(CXX) $(CXXFLAGS) -c main.cpp

# Compiling functions.cpp
functions.o: functions.cpp functions.h
	$(CXX) $(CXXFLAGS) -c functions.cpp

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
