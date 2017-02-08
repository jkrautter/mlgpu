using std::vector;

vector<float> computeRangeBetweenVectors
       (vector<float> base,
        vector<float> vectors,
        float* inverse_covariance_matrix,
        int dim, int n);

vector<float> computeCovarianceMatrix(vector<float> data, int n, int m);

int test();
