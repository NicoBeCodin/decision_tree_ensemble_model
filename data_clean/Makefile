# Compiler
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Files
LIB_SRC = data_clean.cpp
LIB_OBJ = data_clean.o
MAIN_SRC = main_data_clean.cpp
TARGET = main_data_clean

# Phony targets
.PHONY: all clean

# Default target
all: $(TARGET)

# Build the main program
$(TARGET): $(MAIN_SRC) $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(MAIN_SRC) $(LIB_OBJ)

# Compile the library source
$(LIB_OBJ): $(LIB_SRC)
	$(CXX) $(CXXFLAGS) -c $(LIB_SRC) -o $(LIB_OBJ)

# Clean up object files and binaries
clean:
	rm -f $(LIB_OBJ) $(TARGET)
