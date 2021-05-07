CPP=$(wildcard src/*.cpp)
OBJ=$(CPP:.cpp=.o)
DEPS=$(wildcard include/*.h)

NVCC=nvcc

CUDA=$(wildcard src/*.cu)
OBJ_CUDA=$(CUDA:.cu=.o)
DEPS_CUDA=$(wildcard include/*.cuh)
OBJS=$(wildcard src/*.o)


ntt: $(OBJ) $(OBJ_CUDA)
	$(NVCC) -m 64 -o $@ $^ -Iinclude

$(OBJ_CUDA): $(CUDA)
	$(NVCC) -dc -c -o $@ $(@:.o=.cu) -Iinclude	

$(OBJ): $(CPP)
	$(NVCC) -x cu -c -o $@ $(@:.o=.cpp) -Iinclude	

test:
	echo $(CPP)
	echo $(OBJ)
	echo $(OBJ_CUDA)
	echo $(CUDA)

clean:
	rm -rfv ntt $(OBJ) $(OBJ_CUDA)
