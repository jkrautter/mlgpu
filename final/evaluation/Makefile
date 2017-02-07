INC = -I../mahalanobis/ -I../mahalanobis/eigen/eigen-eigen-da9b4e14c255/Eigen

all:
	nvcc  -lcublas $(INC) -rdc=true ../mahalanobis/Utilities.cu -c ../mahalanobis/mahalanobis.cu
	g++ -std=c++11 $(INC) -c evaluation.cpp 
	nvcc -o test mahalanobis.o Utilities.o evaluation.o -L/usr/local/cuda/lib64 -lcublas -lcuda -lcudart
	
	
