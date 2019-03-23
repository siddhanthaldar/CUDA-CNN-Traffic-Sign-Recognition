NVCC = nvcc

cnn: ../include/header.h kernel.cu host.cu
	$(NVCC) -I ./ kernel.cu host.cu -o cnn


run:
	./cnn

clean:
	rm -rf cnn