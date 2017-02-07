#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <Eigen>
#include "mahalanobis.h"
#include "arrayimport.hpp"
#include <string>

using std::vector;
using Eigen::MatrixXf;
using std::endl;
using std::cout;

std::string path = "/home/t2/repr_80354.csv";
std::string covpath = "./InverseCovariance.txt";
const int dim = 50;
const int count = 25000;
Eigen::MatrixXf invCovMat;
bool isComputed = false;

void saveMatrixToFile(float* matrix, int n)
{
	std::fstream f;
	f.open(covpath,std::ios::out);
	for(int i = 0; i < n; i++) f << matrix[i];
//	f.write(reinterpret_cast<char *>(&matrix), n*sizeof(double));
	f.close();
}

std::vector<float> loadMatrix()
{
	std::ifstream is(covpath, std::ios::binary);
  	std::istream_iterator<float> start(is), end;
  	std::vector<float> v(start, end);
	return v;
}

vector<float> LoadData()
{
	float data[dim*count];
	int ids[count];
	arrayimport(data, ids, path.c_str());
        vector<float> v(std::begin(data), std::end(data));
	return v;
}

void computeInvCov()
{
	std::cout << "lade Daten..." << std::endl;        
        vector<float> data = LoadData();
        for(int i = 0; i < 100; i++) std::cout << data.at(i) << " ";
	std::cout <<  std::endl << "..." << std::endl << std::endl;

	std::cout << "berechne Covariance Matrix" << std::endl;
	vector<float> cov = computeCovarianceMatrix(data, count, dim);
        for(int i = 0; i < 100; i++) std::cout << cov.at(i) << " ";
	std::cout <<  std::endl << "..."  << std::endl << std::endl;

	// Use Eigen Matrix Inversion on CPU
	MatrixXf mat = Eigen::Map<MatrixXf>(cov.data(), dim, dim);
	
	std::cout << "invertiere" << std::endl;
	invCovMat = mat.inverse();
	isComputed = true;
        for(int i = 0; i < 100; i++) std::cout << invCovMat.data()[i] << " ";
	std::cout <<  std::endl << "..." << std::endl << std::endl;
	//std::cout << eigenX.inverse() << std::endl;

	saveMatrixToFile(mat.data(), dim*dim);
}

void printVector(vector<float> vec, int start, int end){
        for(int i = start; i < end; i++) cout << std::fixed << std::setprecision(4) << vec.at(i) << " ";
}


vector<float> computeDistances (vector<float> baseVector, vector<float> vectors, int dim, int n)
{
	if(!isComputed) computeInvCov();

	cout << "Basis: " << endl;
	printVector(baseVector, 0, baseVector.size());
	cout << endl << endl;

	vector<float> v = computeRangeBetweenVectors(baseVector, vectors, invCovMat.data(), dim, n);
        
	for(int i = 0; i < 10; i++){
		printVector(vectors, i*dim, (i+1)*dim);
		cout << "Distance: " << v.at(i) << endl << endl;
	}
}


/********/
/* MAIN */
/********/
int main()
{
	//for(int i = 0; i < 10; i++) std::cout << data[i] << "\n";
	
	computeInvCov();
	float* data = invCovMat.data();
        vector<float> base(data + 0*dim, data+1*dim);
	cout << "Size Basis: " << base.size() << endl << endl;

        vector<float> vect(data + 0*dim, data+20*dim);
	cout << "Size Vectors: " << vect.size() << endl << endl;
	computeDistances(base, vect, dim, 20);
}
