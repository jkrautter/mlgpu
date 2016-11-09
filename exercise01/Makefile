all:
	nvcc -D_FORCE_INLINES -O3 -g --compiler-options -O3,-g,-Wall,-Wno-unused-result mnist-knn.cu -lm -o mnist-knn
clean:
	rm -f mnist-knn
