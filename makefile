TARGET=main
OBJECTS=main.o util.o matmul.o ver3.o ver4.o ver5.o

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
CUDA_CFLAGS=-std=c++14 -O3 -Xcompiler -Wall -Xcompiler -march=native -Xcompiler -mavx2 -Xcompiler -mfma -Xcompiler -fopenmp -Xcompiler -mno-avx512f -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

CXX=g++
NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

ver3.o: ver3.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $^

ver4.o: ver4.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $^

ver5.o: ver5.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $^

# matmul.o 생성
matmul.o: matmul.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
