// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

// raisimGymForRainbow include
#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "PogoController.hpp"

namespace raisim {

    class ENVIRONMENT {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
                resourceDir_(resourceDir), visualizable_(visualizable) {
            setSeed(id);

            pogo_ = world_.addArticulatedSystem(resourceDir_+"/pogo/urdf/pogo.urdf");
            pogo_->setName("pogo");

            pogo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            auto* ground = world_.addGround();

            /// create controller
            controller_.create(&world_);

            /// set curriculum
            simulation_dt_ = controller_.getSimDt();
            control_dt_ = controller_.getConDt();
            READ_YAML(double, cmdcurriculumFactor_, cfg["curriculum"]["cmd_initial_factor"])
            READ_YAML(double, cmdcurriculumDecayFactor_, cfg["curriculum"]["cmd_decay_factor"])

            /// get robot data
            gcDim_ = pogo_->getGeneralizedCoordinateDim();
            gvDim_ = pogo_->getDOF();

            /// initialize containers
            gc_init_.setZero(gcDim_);
            gv_init_.setZero(gvDim_);
            gc_init_from_.setZero(gcDim_);
            gv_init_from_.setZero(gvDim_);
            gc_init_head_.setZero(7);

            gc_init_head_ << 0, 0, 1.0, 1.0, 0.0, 0.0, 0.0;
            gc_init_.head(7) << gc_init_head_;

            gc_init_.tail(nJoints_).setConstant(0.0);
            gc_init_from_ = gc_init_;
            pogo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            // Reward coefficients
            controller_.setRewardConfig(cfg);

            if(visualizable_){
                server_ = std::make_unique<raisim::RaisimServer>(&world_);
//                commandArrow_ = server_->addVisualArrow("command arrow", 0.2, 3.0, 1.0);
//                gvArrow_ = server_->addVisualArrow("gv arrow", 0.2, 3.0, 0.0, 1.0);
                server_->launchServer();

            }

        }

        ~ENVIRONMENT() { if (server_) server_->killServer(); }
        void init () { }
        void close () { }
        void setSimulationTimeStep(double dt) {
            world_.setTimeStep(dt);
            controller_.setSimDt(dt);
            simulation_dt_ = controller_.getSimDt();
        };
        void setControlTimeStep(double dt) {
            controller_.setConDt(dt);
            control_dt_ = controller_.getConDt();
        };
        void turnOffVisualization() { server_->hibernate(); }
        void turnOnVisualization() { server_->wakeup(); }
        void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
        void stopRecordingVideo() { server_->stopRecordingVideo(); }
        const std::vector<std::string>& getStepDataTag() { return controller_.getStepDataTag(); }
        const Eigen::VectorXd& getStepData() { return controller_.getStepData(); }

        void reset() {
            gc_init_.head(7) << gc_init_head_;
            gc_init_.tail(nJoints_).setConstant(0.0);

            /// randomize orientation
            raisim::Mat<3,3> rotMat, yawRot, pitchRollMat;
            raisim::Vec<4> quaternion;
            raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
            axis /= axis.norm();
            raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.02, pitchRollMat);
            raisim::angleAxisToRotMat({0,0,1}, uniDist_(gen_) * 2. * M_PI, yawRot);
            rotMat = pitchRollMat * yawRot;
            raisim::rotMatToQuat(rotMat, quaternion);
            gc_init_.segment(3, 4) = quaternion.e();

            /// randomize gc & gv
            for(int i=0 ; i<nJoints_; i++)
                gc_init_[i+7] = 0.2 * std::clamp(normDist_(gen_), -1.0, 1.0);

            Eigen::VectorXd jointVel(nJoints_);
            for(int i=0; i<nJoints_; i++) jointVel[i] = 0.5 * std::clamp(normDist_(gen_), -1.0, 1.0);
            gv_init_.tail(nJoints_) << jointVel;


            /// command sampling
            command_.setZero(3);
            if(cmdcurriculumFactor_ > 0.2){
                if(uniDist_(gen_) < 0.2) {
                    command_.setZero(3);
                } else {
                    do {
                        command_ << (maxSpeed_* (2*uniDist_(gen_) - 1.0)) * cmdcurriculumFactor_, /// ~ U(-maxSpeed, maxSpeed)
                                (maxSpeed_* (2*uniDist_(gen_) - 1.0)) * cmdcurriculumFactor_, /// ~ U(-maxSpeed, maxSpeed)
                                (2*uniDist_(gen_) - 1.0) * cmdcurriculumFactor_; /// ~ U(-1, 1)

                        double p = uniDist_(gen_);
                        if (p < 1. / 8.) command_ << command_(0), 0., 0.;
                        else if (p < 2. / 8.) command_ << 0., command_(1), 0.0;
                        else if (p < 3. / 8.) command_ << command_(0), command_(1), 0.0;
                        else if (p < 4. / 8.) command_ << 0., 0., command_(2);
                    } while (command_.norm() < 0.2);
                }
            }


            pogo_->setState(gc_init_, gv_init_);
            controller_.reset(gen_, normDist_);

            controller_.updateStateVariables(heightMap_, gen_, normDist_);
        }


        double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
            /// action scaling
            controller_.advance(action);
            float dummy;
            int howManySteps;

            for(howManySteps = 0; howManySteps < int(control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {
                subStep(visualize);

                if(isTerminalState(dummy)) {
                    howManySteps++;
                    break;
                }
            }
            return controller_.getRewardSum(howManySteps, cmdcurriculumFactor_);
        }

        void setCommand(const Eigen::Ref<EigenVec>& command) { command_ = command.cast<double>();}

        void subStep(bool visualize) {
            if(server_) server_->lockVisualizationServerMutex();
            world_.integrate();
            if(server_) server_->unlockVisualizationServerMutex();

            controller_.updateStateVariables(heightMap_, gen_, normDist_);
            controller_.accumulateRewards(command_, 1.0);

            if (visualizable_ && visualize)
                raisim::MSLEEP(2);
        }

        void observe(Eigen::Ref<EigenVec> ob) {
            controller_.updateObservation(command_);
            controller_.getObservation(obScaled_);
            ob = obScaled_.cast<float>();
        }

        void getValueObs(Eigen::Ref<EigenVec> valueOb) {
            controller_.updateValueObservation(command_);
            controller_.getValueObservation(valueObScaled_);
            valueOb = valueObScaled_.cast<float>();
        }

        bool isTerminalState(float& terminalReward) {
            return controller_.isTerminalState(terminalReward);
        }

        void setSeed(int seed) {
            gen_.seed(seed);
        }

        void curriculumUpdate() {
            cmdcurriculumFactor_ = cmdcurriculumFactor_ + cmdcurriculumDecayFactor_;
            cmdcurriculumFactor_ = std::min(cmdcurriculumFactor_, 1.0);
        }

        static constexpr int getObDim() { return RaiboController::getObDim(); }
        static constexpr int getValueObDim() { return RaiboController::getValueObDim(); }
        static constexpr int getActionDim() { return RaiboController::getActionDim(); }
        int getGroundNum() { return 1; }

        void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
            controller_.getState(gc, gv);
        }

        void getLoggingInfo(Eigen::Ref<EigenVec> info) {
            controller_.getLoggingInfo(command_, info);
        }

        void visualizeArrow(){
//            commandArrow_->setPosition(controller_.getArrowPosition());
//            commandArrow_->setOrientation(controller_.getCommandArrowOrientation(command_));
//
//            gvArrow_->setPosition(controller_.getArrowPosition());
//            gvArrow_->setOrientation(controller_.getgvArrowOrientation());
        }

    protected:
        static constexpr int nJoints_ = 3;
        raisim::World world_;
        double simulation_dt_;
        double control_dt_;
        int gcDim_, gvDim_;
        const std::string resourceDir_;

        raisim::ArticulatedSystem* pogo_;
        raisim::HeightMap* heightMap_;
        Eigen::VectorXd gc_init_, gv_init_, gc_init_head_;
        Eigen::VectorXd gc_init_from_, gv_init_from_;

        double cmdcurriculumFactor_, cmdcurriculumDecayFactor_;
        Eigen::VectorXd obScaled_, valueObScaled_;
        Eigen::Vector3d command_;
        bool visualizable_ = false;
        RaiboController controller_;
        double maxSpeed_ = 1.5;

        std::unique_ptr<raisim::RaisimServer> server_;
        raisim::Visuals *commandArrow_, *gvArrow_;

        thread_local static std::mt19937 gen_;
        thread_local static std::normal_distribution<double> normDist_;
        thread_local static std::uniform_real_distribution<double> uniDist_;

    };

    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
    thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
    thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);

}