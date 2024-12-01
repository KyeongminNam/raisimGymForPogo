//e
// Created by Jemin Hwangbo on 2022/04/08.
//

#define _MAKE_STR(x) __MAKE_STR(x)
#define __MAKE_STR(x) #x
#include "raisim/RaisimServer.hpp"
#include "random_coordinates.hpp"

size_t TOTAL_STEPS = 20000;
size_t SPRING_GC_IDX = 1; // this is a 1D version. Usually the spring gc idx is 8
double PRELOAD_N = 1000;
double SPRING_CONST_N_M = 15000;

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  
  // create raisim world
  raisim::World world; // physics world
  raisim::RaisimServer server(&world);

  auto pogo = world.addArticulatedSystem(std::string(_MAKE_STR(RESOURCE_DIR)) + "/pogo/urdf/pogo_1D.urdf");
  // pogo -> setComputeInverseDynamics(true);
  // std::cout << "robot was loaded!" << std::endl;

  world.addGround();
  world.setTimeStep(0.001);

  Eigen::VectorXd gc(pogo->getGeneralizedCoordinateDim()), gv(pogo->getDOF());

  gc << 
    0.5, 0.2, 0.0;

  gv << 
    0.0, 0.0, 2.0;


  // Declare variables (should be in private section)
  int gcDim_, gvDim_, nJoints_, playerNum_ = 1;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  int obDim_ = 0, actionDim_ = 0;

  gcDim_ = pogo->getGeneralizedCoordinateDim();
  gvDim_ = pogo->getDOF();
  // nJoints_ = gvDim_ - 6;
  nJoints_ = 1; // for 1D version

  /// initialize containers
  gc_.setZero(gcDim_);
  gc_init_.setZero(gcDim_);
  gv_.setZero(gvDim_);
  gv_init_.setZero(gvDim_);
  pTarget_.setZero(gcDim_);
  vTarget_.setZero(gvDim_);
  pTarget12_.setZero(nJoints_);

  /// set pd gains
  Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
  jointPgain.setZero();
  jointPgain.tail(nJoints_).setConstant(50.0);
  jointDgain.setZero();
  jointDgain.tail(nJoints_).setConstant(0.2);

  jointPgain.tail(1) << 15000.0; //joint P gain for prismatic joint (N/m)

  // setting spring / preload
  jointPgain[SPRING_GC_IDX] = SPRING_CONST_N_M;
  pTarget_[SPRING_GC_IDX] = -(PRELOAD_N / SPRING_CONST_N_M);

  pogo->setPdGains(jointPgain, jointDgain);
  pogo->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

  // utils::gcRandomize(gc);
  // gc[2] = gc[2] + 3;
  // utils::gvRandomize(gv,15);

  pogo->setState(gc, gv);
  server.launchServer();
  server.focusOn(pogo);

  /// if you are using an old version of Raisim, you need this line
  world.integrate1();

  for (int sec=3; sec>0; sec--){
    std::cout << "Starting in [" << sec << "]..." << std::endl;
    raisim::USLEEP(1000000);
  }
  std::cout << "START!" << std::endl;

  // SIM LOOP

  // diagnostic variables
  double z;
  raisim::Vec<3> pos;

  // controller variables
  int latch = 0;

  for (size_t t = 0; t<TOTAL_STEPS; t++){
    RS_TIMED_LOOP(world.getTimeStep()*2e6)
    server.integrateWorldThreadSafe();
    pogo->getState(gc, gv);
    
    // analyze step here
    std::cout<<"STEP " << t << "/" << TOTAL_STEPS << std::endl;
    
    z = pogo->getBodyCOM_W()[2][2];
    std::cout << "  CoM Height: " << z*1000 << "mm" << std::endl;
    pogo->getFramePosition("tip_foot_fixed",pos);
    std::cout << "  Foot Pos  : " << pos[2] << std::endl;

    // set pd targets here
    // pTarget_.tail(1) << ((t/3000)%2 == 0) * 0.4; // some random periodic target

    // simplest reactive targeting
    // keep in mind, that this will control the base to fall with exactly 0.1m/s at the apex !
    if(gv[0] < -0.1){pTarget_.tail(1) << 0.0;}
    else{pTarget_.tail(1) << 0.4;}

    pogo->setPdTarget(pTarget_,vTarget_);
  }

  server.killServer();

  std::cout<<"SIMULATION COMPLETE"<<std::endl;

  return 0;
}
