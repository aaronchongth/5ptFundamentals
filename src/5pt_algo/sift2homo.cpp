#include "sift2homo.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

bool sift_to_homography(const Mat& points_1, const Mat& points_2,
                        const Mat& angles, Mat& homography) {
  // construct matrix A, to find Ah = 0
  Mat A(6, 9, CV_64F);
  for (int i = 0; i < 3; i++) {
    double x_1 = points_1.at<double>(i, 0);
    double y_1 = points_1.at<double>(i, 1);
    double x_2 = points_2.at<double>(i, 0);
    double y_2 = points_2.at<double>(i, 1);

    A.at<double>(i, 0) = -x_1;
    A.at<double>(i, 1) = -y_1;
    A.at<double>(i, 2) = -1;
    A.at<double>(i, 3) = 0;
    A.at<double>(i, 4) = 0;
    A.at<double>(i, 5) = 0;
    A.at<double>(i, 6) = x_1 * x_2;
    A.at<double>(i, 7) = x_2 * y_1;
    A.at<double>(i, 8) = x_2;

    A.at<double>(i + 1, 0) = 0;
    A.at<double>(i + 1, 1) = 0;
    A.at<double>(i + 1, 2) = 0;
    A.at<double>(i + 1, 3) = -x_1;
    A.at<double>(i + 1, 4) = -y_1;
    A.at<double>(i + 1, 5) = -1;
    A.at<double>(i + 1, 6) = x_1 * y_2;
    A.at<double>(i + 1, 7) = y_1 * y_2;
    A.at<double>(i + 1, 8) = y_2;
  }
  A = A / norm(A);

  // there is 9 - 6 = 3 dof left
  // H = alpha * h1 + beta * h2 + h3
  Mat W(6, 1, CV_64F);
  Mat U(6, 9, CV_64F);
  Mat Vt(9, 6, CV_64F);
  SVDecomp(A, W, U, Vt, SVD::MODIFY_A + SVD::FULL_UV);
  Mat N = Vt(Range(0, 9), Range(3, 6));
  N = N / norm(N);
  double n11 = N.at<double>(0, 0);
  double n12 = N.at<double>(1, 0);
  double n13 = N.at<double>(2, 0);
  double n14 = N.at<double>(3, 0);
  double n15 = N.at<double>(4, 0);
  double n16 = N.at<double>(5, 0);
  double n17 = N.at<double>(6, 0);
  double n18 = N.at<double>(7, 0);
  double n19 = N.at<double>(8, 0);

  double n21 = N.at<double>(0, 1);
  double n22 = N.at<double>(1, 1);
  double n23 = N.at<double>(2, 1);
  double n24 = N.at<double>(3, 1);
  double n25 = N.at<double>(4, 1);
  double n26 = N.at<double>(5, 1);
  double n27 = N.at<double>(6, 1);
  double n28 = N.at<double>(7, 1);
  double n29 = N.at<double>(8, 1);

  double n31 = N.at<double>(0, 3);
  double n32 = N.at<double>(1, 3);
  double n33 = N.at<double>(2, 3);
  double n34 = N.at<double>(3, 3);
  double n35 = N.at<double>(4, 3);
  double n36 = N.at<double>(5, 3);
  double n37 = N.at<double>(6, 3);
  double n38 = N.at<double>(7, 3);
  double n39 = N.at<double>(8, 3);

  // estimate homography parameters
  int p[] = {0, 1, 2};
  sort(p, p + 3);
  double best_error = DBL_MAX;
  Mat best_hom;
  while (next_permutation(p, p + 3)) {
    int m = p[0];
    int n = p[1];
    double a1 = angles.at<double>(m, 0);
    double a2 = angles.at<double>(n, 0);

    double x11 = points_1.at<double>(m, 0);
    double y11 = points_1.at<double>(m, 1);
    double x21 = points_1.at<double>(n, 0);
    double y21 = points_1.at<double>(n, 1);
    double x12 = points_2.at<double>(m, 0);
    double y12 = points_2.at<double>(m, 1);
    double x22 = points_2.at<double>(n, 0);
    double y22 = points_2.at<double>(n, 1);

    double x = (cos(a2) * cos(a1) * n24 * n37 * y12 -
                cos(a2) * cos(a1) * n24 * n37 * y22 -
                cos(a2) * cos(a1) * n27 * n34 * y12 +
                cos(a2) * cos(a1) * n27 * n34 * y22 +
                cos(a2) * n21 * n37 * y22 * sin(a1) -
                cos(a2) * n24 * n37 * x12 * sin(a1) -
                cos(a2) * n27 * n31 * y22 * sin(a1) +
                cos(a2) * n27 * n34 * x12 * sin(a1) -
                cos(a1) * n21 * n37 * y12 * sin(a2) +
                cos(a1) * n24 * n37 * x22 * sin(a2) +
                cos(a1) * n27 * n31 * y12 * sin(a2) -
                cos(a1) * n27 * n34 * x22 * sin(a2) +
                n21 * n37 * x12 * sin(a2) * sin(a1) -
                n21 * n37 * x22 * sin(a2) * sin(a1) -
                n27 * n31 * x12 * sin(a2) * sin(a1) +
                n27 * n31 * x22 * sin(a2) * sin(a1) -
                cos(a2) * n21 * n34 * sin(a1) + cos(a2) * n24 * n31 * sin(a1) +
                cos(a1) * n21 * n34 * sin(a2) - cos(a1) * n24 * n31 * sin(a2)) /
               (-cos(a2) * n11 * n24 * sin(a1) + cos(a2) * n14 * n21 * sin(a1) +
                cos(a1) * n11 * n24 * sin(a2) - cos(a1) * n14 * n21 * sin(a2) +
                n11 * n27 * x12 * sin(a1) * sin(a2) +
                n17 * n21 * x22 * sin(a1) * sin(a2) +
                cos(a2) * cos(a1) * n14 * n27 * y12 -
                cos(a2) * cos(a1) * n14 * n27 * y22 -
                cos(a2) * cos(a1) * n17 * n24 * y12 +
                cos(a2) * cos(a1) * n17 * n24 * y22 +
                cos(a2) * n11 * n27 * y22 * sin(a1) -
                cos(a2) * n14 * n27 * x12 * sin(a1) -
                cos(a2) * n17 * n21 * y22 * sin(a1) +
                cos(a2) * n17 * n24 * x12 * sin(a1) -
                n11 * n27 * x22 * sin(a1) * sin(a2) -
                n17 * n21 * x12 * sin(a1) * sin(a2) -
                cos(a1) * n11 * n27 * y12 * sin(a2) +
                cos(a1) * n14 * n27 * x22 * sin(a2) +
                cos(a1) * n17 * n21 * y12 * sin(a2) -
                cos(a1) * n17 * n24 * x22 * sin(a2));
    double y =
        -(cos(a1) * n14 * n37 * x22 * sin(a2) +
          cos(a1) * n17 * n31 * y12 * sin(a2) -
          cos(a1) * n17 * n34 * x22 * sin(a2) -
          n11 * n37 * x22 * sin(a1) * sin(a2) -
          n17 * n31 * x12 * sin(a1) * sin(a2) +
          n11 * n37 * x12 * sin(a1) * sin(a2) +
          n17 * n31 * x22 * sin(a1) * sin(a2) +
          cos(a2) * cos(a1) * n14 * n37 * y12 -
          cos(a2) * cos(a1) * n14 * n37 * y22 -
          cos(a2) * cos(a1) * n17 * n34 * y12 +
          cos(a2) * cos(a1) * n17 * n34 * y22 +
          cos(a2) * n11 * n37 * y22 * sin(a1) -
          cos(a2) * n14 * n37 * x12 * sin(a1) -
          cos(a2) * n17 * n31 * y22 * sin(a1) +
          cos(a2) * n17 * n34 * x12 * sin(a1) -
          cos(a1) * n11 * n37 * y12 * sin(a2) - cos(a2) * n11 * n34 * sin(a1) +
          cos(a2) * n14 * n31 * sin(a1) + cos(a1) * n11 * n34 * sin(a2) -
          cos(a1) * n14 * n31 * sin(a2)) /
        (-cos(a2) * n11 * n24 * sin(a1) + cos(a2) * n14 * n21 * sin(a1) +
         cos(a1) * n11 * n24 * sin(a2) - cos(a1) * n14 * n21 * sin(a2) +
         n11 * n27 * x12 * sin(a1) * sin(a2) +
         n17 * n21 * x22 * sin(a1) * sin(a2) +
         cos(a2) * cos(a1) * n14 * n27 * y12 -
         cos(a2) * cos(a1) * n14 * n27 * y22 -
         cos(a2) * cos(a1) * n17 * n24 * y12 +
         cos(a2) * cos(a1) * n17 * n24 * y22 +
         cos(a2) * n11 * n27 * y22 * sin(a1) -
         cos(a2) * n14 * n27 * x12 * sin(a1) -
         cos(a2) * n17 * n21 * y22 * sin(a1) +
         cos(a2) * n17 * n24 * x12 * sin(a1) -
         n11 * n27 * x22 * sin(a1) * sin(a2) -
         n17 * n21 * x12 * sin(a1) * sin(a2) -
         cos(a1) * n11 * n27 * y12 * sin(a2) +
         cos(a1) * n14 * n27 * x22 * sin(a2) +
         cos(a1) * n17 * n21 * y12 * sin(a2) -
         cos(a1) * n17 * n24 * x22 * sin(a2));

    Mat h = x * N.col(0) + y * N.col(1) + N.col(2);
    Mat Ht = h.reshape(1, 3);
    Mat H = Ht.t();

    int k = p[2];
    double x1 = points_1.at<double>(k, 0);
    double y1 = points_1.at<double>(k, 1);
    double x2 = points_2.at<double>(k, 0);
    double y2 = points_2.at<double>(k, 1);
    Mat A(2, 2, CV_64F);
    homography_to_affine(H, x1, y1, x2, y2, A);
    // if (homography_to_affine(H, x1, y1, x2, y2, A))
    //   cout << "Homography to affine done." << endl;

    // decompose affine
    double sx, sy, alpha, w;
    decompose_affine(A, sx, sy, alpha, w);
    // if (decompose_affine(A, sx, sy, alpha, w))
    //   cout << "Affine decomposed." << endl;

    // find the error, and try to update
    double curr_error = abs(alpha - angles.at<double>(k));
    if (curr_error < best_error) {
      best_error = curr_error;
      best_hom = H;
    }
  }

  // normalize the best homography
  homography = best_hom / best_hom.at<double>(2, 2);
  return true;
}

bool homography_to_affine(const Mat& H, double x1, double y1, double x2,
                          double y2, Mat& A) {
  Mat new_point(3, 1, CV_64F);
  new_point.at<double>(0, 0) = x1;
  new_point.at<double>(1, 0) = y1;
  new_point.at<double>(2, 0) = 1;

  Mat S = H.row(2) * new_point;
  double s = S.at<double>(0, 0);

  double a11 = (H.at<double>(0, 0) - H.at<double>(2, 0) * x2) / s;
  double a12 = (H.at<double>(0, 1) - H.at<double>(2, 1) * x2) / s;
  double a21 = (H.at<double>(1, 0) - H.at<double>(2, 0) * y2) / s;
  double a22 = (H.at<double>(1, 1) - H.at<double>(2, 1) * y2) / s;

  A.at<double>(0, 0) = a11;
  A.at<double>(0, 1) = a12;
  A.at<double>(1, 0) = a21;
  A.at<double>(1, 1) = a22;
  return true;
}

bool decompose_affine(const Mat& A, double& sx, double& sy, double& alpha,
                      double& w) {
  alpha = atan2(A.at<double>(1, 0), A.at<double>(0, 0));
  sx = A.at<double>(0, 0) / cos(alpha);
  sy = (cos(alpha) * A.at<double>(1, 1) - sin(alpha) * A.at<double>(0, 1)) /
       (cos(alpha) * cos(alpha) + sin(alpha) * sin(alpha));
  w = (A.at<double>(0, 1) + sin(alpha) * sy) / cos(alpha);
  return true;
}
