# 定义编译器
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# 目标文件名
TARGET = test_fonctions_io

# 源文件
SRC = test_functions_io.cpp functions_io.cpp

# 默认目标，编译可执行文件
all: $(TARGET)

# 编译可执行文件
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# 清理目标文件和临时文件
clean:
	rm -f $(TARGET)
