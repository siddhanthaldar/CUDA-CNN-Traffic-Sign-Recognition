NVCC = nvcc

cnn: header.h kernel.cu host.cu
	$(NVCC) -I ./ kernel.cu host.cu -o cnn


run:
	./cnn

clean:
	rm -rf cnn