//e
// Created by Jemin Hwangbo on 2022/04/08.
//

#define _MAKE_STR(x) __MAKE_STR(x)
#define __MAKE_STR(x) #x
#include "raisim/RaisimServer.hpp"
#include "random_coordinates.hpp"

// START HEADER FILE

// HEADER END

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  // create raisim world
  raisim::World world; // physics world
  raisim::RaisimServer server(&world);

  auto pogo = world.addArticulatedSystem(std::string(_MAKE_STR(RESOURCE_DIR)) + "/pogo/urdf/pogo.urdf");

  // std::cout << "robot was loaded!" << std::endl;

  world.addGround();
  world.setTimeStep(0.001);

  Eigen::VectorXd gc(pogo->getGeneralizedCoordinateDim()), gv(pogo->getDOF());

  gc << 
    0, 0, 0.54, 
    1.0, 0.0, 0.0, 0.0,
    0.1, 0.0, 0.0, 0.0;

  gv << 
    0.0, 0.0, 3.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0;

  // utils::gcRandomize(gc);
  // gc[2] = gc[2] + 3;
  // utils::gvRandomize(gv,15);

  pogo->setState(gc, gv);
  server.launchServer();
  server.focusOn(pogo);

  /// if you are using an old version of Raisim, you need this line
  world.integrate1();

  for (int sec=3; sec>0; sec--){
    std::cout << "Dropping in [" << sec << "]..." << std::endl;
    raisim::USLEEP(1000000);
  }
  std::cout << "DROP!" << std::endl;

  // SIM LOOP
  for (size_t i = 0; i<20000; i++){
    RS_TIMED_LOOP(world.getTimeStep()*2e6)
    server.integrateWorldThreadSafe();
    pogo->getState(gc, gv);
  }

  server.killServer();

  std::cout<<"SIMULATION COMPLETE"<<std::endl;

  return 0;
}
