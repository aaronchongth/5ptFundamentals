#include "sift2homo.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

bool sift_to_homography(const vector<KeyPoint>& img_1_pts,
                        const vector<KeyPoint>& img_2_pts,
                        const vector<double>& angles, MatrixXd& homography) {
  // check for 3 points
  if (img_1_pts.size() != 3 || img_2_pts.size() != 3 || angles.size() != 3)
    cry("Pass 3 points from each image into sift_to_homography please.");

  // construct matrix A, to find Ah = 0
  MatrixXf A(6, 9);
  for (int i = 0; i < 3; i++) {
    float x_1 = img_1_pts[i].pt.x;
    float y_1 = img_1_pts[i].pt.y;
    float x_2 = img_2_pts[i].pt.x;
    float y_2 = img_2_pts[i].pt.y;

    A.row(i) << -x_1, -y_1, -1, 0, 0, 0, x_1 * x_2, x_2 * y_1, x_2;
    A.row(i + 1) << 0, 0, 0, -x_1, -y_1, -1, x_1 * y_2, y_1 * y_2, y_2;
  }
  cout << A << endl;

  // null space
  // FullPivLU<MatrixXd> lu(A);
  // MatrixXd A_null_space = lu.kernel();

  // // null space test
  // MatrixXf m = MatrixXf::Random(3, 5);
  // cout << "Here is the matrix m:" << endl << m << endl;
  // MatrixXf ker = m.fullPivLu().kernel();
  // cout << "Here is a matrix whose columns form a basis of the kernel of m:"
  //      << endl
  //      << ker << endl;
  // cout << "By definition of the kernel, m*ker is zero:" << endl
  //      << m * ker << endl;

  cout << "All done." << endl;
  return true;
}