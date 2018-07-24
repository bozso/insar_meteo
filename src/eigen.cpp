#include <iostream>
#include <Eigen/Dense>

#define SIZE 2500

using Eigen::MatrixXd;
using namespace std;

int main()
{
    MatrixXd a(SIZE,SIZE);
    MatrixXd b(SIZE,SIZE);
    MatrixXd c(SIZE,SIZE);

    for(uint ii = 0; ii < a.rows(); ii++) {
        for(uint jj = 0; jj < a.cols(); jj++) {
            a(ii, jj) = ii + jj;
            b(ii, jj) = jj + ii;
        }
    }
    c = a * b;
    
    cout << c.block(0,0,1,10) << endl;
}
