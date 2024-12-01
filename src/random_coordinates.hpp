#ifndef ME553_UTILS_RANDOM_COORDINATES_HPP_
#define ME553_UTILS_RANDOM_COORDINATES_HPP_
#include <Eigen/Core>

namespace utils{

void gcRandomize(Eigen::VectorXd& gc,double gamma = 1){
    int dims = gc.size();

    std::srand((unsigned int) time(nullptr));
    Eigen::VectorXd noise = Eigen::VectorXd::Random(dims);

    Eigen::Vector4d quat_raw = Eigen::Vector4d::Random(4);
    Eigen::Vector3d quat_xyz = quat_raw.segment(1,3)/quat_raw.segment(1,3).norm();
    double quat_angle = quat_raw(0)*M_PI;
    Eigen::Vector4d quat_rand;
    quat_rand << cos(quat_angle/2),sin(quat_angle/2)*quat_xyz;
    // std::cout << "random quaternion" << quat_rand.transpose() << " (norm: " << quat_rand.norm() << ")" << std::endl;
    // std::cout << "quat axis" << quat_xyz.transpose() << " (norm: " << quat_xyz.norm() << ")"  << std::endl;
    // std::cout << "quat raw" << quat_raw.transpose() << " (norm: " << quat_raw.norm() << ")"  << std::endl;

    gc = gc+gamma*noise;
    gc(3) = quat_rand(0);
    gc(4) = quat_rand(1);
    gc(5) = quat_rand(2);
    gc(6) = quat_rand(3);
}

void gvRandomize(Eigen::VectorXd& gv, double gamma = 1){
    int dims = gv.size();

    std::srand((unsigned int) time(nullptr));
    Eigen::VectorXd noise = Eigen::VectorXd::Random(dims);

    gv = gv+gamma*noise;
}

}

#endif
