//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_pogo_CONTROLLER_HPP
#define _RAISIM_GYM_pogo_CONTROLLER_HPP

namespace raisim {

    class RaiboController {
    public:
        inline bool create(raisim::World *world) {
            pogo_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("pogo"));
            gcDim_ = pogo_->getGeneralizedCoordinateDim();
            gvDim_ = pogo_->getDOF();
            gc_.setZero(gcDim_);
            gv_.setZero(gvDim_);

            /// action
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            actionScaled_.setZero(actionDim_);
            previousAction_.setZero(actionDim_);
            prevprevAction_.setZero(actionDim_);
            prepreprevAction_.setZero(actionDim_);
            actionDoulbe_.setZero(actionDim_);
            obDouble_.setZero(obDim_);
            valueObDouble_.setZero(valueObDim_);
            actionStd_.setConstant(0.3);

            /// pd controller
            jointPgain_.setZero(gvDim_); jointDgain_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);

            jointPgain_.tail(nJoints_).setConstant(pGainRev_);
            jointPgain_[6 + POGO_GC_PRISMATIC_IDX] = pGainPrsm_;
            jointPgain_[6 + POGO_GC_PASSIVE_IDX] = pogoSpringConstant_;
            pTarget_[7 + POGO_GC_PASSIVE_IDX] = -(pogoPreload_ / pogoSpringConstant_);
            jointDgain_.tail(nJoints_).setConstant(0.2);

            pogo_->setPdGains(jointPgain_, jointDgain_);
            pogo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// state data
            jointVelocity_.resize(nJoints_);
            footIndex_.push_back(pogo_->getBodyIdx("tip"));
            smoothingWeight_.setZero(actionDim_);
            smoothingWeight_ << 4., 4., 1.;

            /// exported data
            stepDataTag_ = {"command_tracking_rew",
                            "torque_rew",
                            "joint_vel_rew",
                            "smooth_rew",
                            "smooth_rew2",
                            "orientation_rew",
                            "con_rew",
                            "base_motion_rew",
                            "clearance_rew",
                            "base_height_limit_rew",
                            "airtime_rew",
                            "orientation_rew2",
                            "positive_rew",
                            "negative_rew",
                            "curriculum"};
            stepData_.resize(stepDataTag_.size());

            return true;
        };

        void updateStateVariables(const raisim::HeightMap *map, std::mt19937 &gen_,
                                  std::normal_distribution<double> &normDist_) {
            pogo_->getState(gc_, gv_);
            jointVelocity_ = gv_.tail(nJoints_);
            baseRot_ = pogo_->getBaseOrientation();
            pogo_->getBodyOrientation(pogo_->getBodyIdx("mass"), baseRot2_);

            bodyLinVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
            bodyLinVel2_ = baseRot2_.e().transpose() * gv_.segment(0, 3);
            bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);
            bodyAngVel2_ = baseRot2_.e().transpose() * gv_.segment(3, 3);
            baseHeight_ = gc_[2];

            /// check if the feet are in contact with the ground
            footContactState_ = false;
            for (auto &contact: pogo_->getContacts()) {
                if (contact.skip()) continue;
                auto it = std::find(footIndex_.begin(), footIndex_.end(), contact.getlocalBodyIndex());
                size_t index = it - footIndex_.begin();
                if (index < 1 && !contact.isSelfCollision()){
                    footContactState_ = true;
                }
            }

            /// airtime & standtime
            if (footContactState_) {
                airTime_ = 0;
                stanceTime_ += simDt_;
            } else {
                airTime_ += simDt_;
                stanceTime_ = 0;
            }

        }

        void getObservation(Eigen::VectorXd &observation) {
            observation = obDouble_;
        }

        void getValueObservation(Eigen::VectorXd &valueObservation) {
            valueObservation = valueObDouble_;
        }

        bool advance(const Eigen::Ref<EigenVec> &action) {
            /// action scaling
            actionDoulbe_ = action.cast<double>();
            actionDoulbe_ = actionDoulbe_.cwiseProduct(actionStd_);
            actionDoulbe_ += actionMean_;

            pTarget_.tail(nJoints_) = actionDoulbe_;
            pogo_->setPdTarget(pTarget_, vTarget_);

            prepreprevAction_ = prevprevAction_;
            prevprevAction_ = previousAction_;
            previousAction_ = actionDoulbe_;
            return true;
        }

        void reset(std::mt19937 &gen_,
                   std::normal_distribution<double> &normDist_) {
            pogo_->getState(gc_, gv_);
            actionDoulbe_.setZero();
            previousAction_ << gc_.tail(nJoints_);
            prevprevAction_ << gc_.tail(nJoints_);
            prepreprevAction_ << gc_.tail(nJoints_);
        }

        [[nodiscard]] float getRewardSum(const int & howManySteps, double cf) {
            double positiveReward, negativeReward, totalReward;
            stepData_[0] = commandTrackingReward_;
            stepData_[1] = torqueReward_;
            stepData_[2] = jointVelocityReward_;
            stepData_[3] = smoothReward_;
            stepData_[4] = smoothReward2_;
            stepData_[5] = orientationReward_;
            stepData_[6] = conReward_;
            stepData_[7] = basemotionReward_;
            stepData_[8] = clearanceReward_;
            stepData_[9] = baseheightLimitReward_;
            stepData_[10] = airtimeReward_;
            stepData_[11] = orientationReward2_;
            stepData_[12] = 0.0;
            stepData_[13] = 0.0;
            stepData_[14] = 0.0;
            stepData_ /= howManySteps;
            totalReward = stepData_.sum();
            positiveReward = stepData_[0] + stepData_[5] + stepData_[10]+ stepData_[11];
            negativeReward = totalReward - positiveReward;
            stepData_[12] = positiveReward;
            stepData_[13] = negativeReward;
            stepData_[14] = cf;

            commandTrackingReward_ = 0.;
            torqueReward_ = 0.;
            jointVelocityReward_ = 0.;
            smoothReward_ = 0.;
            smoothReward2_ = 0.;
            orientationReward_ = 0.;
            conReward_ = 0.;
            basemotionReward_ = 0.;
            clearanceReward_ = 0.;
            baseheightLimitReward_ = 0.;
            airtimeReward_ = 0.;
            orientationReward2_ = 0.;

            return float(positiveReward * std::exp(0.2 * negativeReward));
        }
        [[nodiscard]] bool isTerminalState(float &terminalReward) {
            terminalReward = float(terminalReward_);

            /// if the contact body is not feet
            for (auto &contact: pogo_->getContacts()) {
                    if(contact.getlocalBodyIndex() == pogo_->getBodyIdx("mass")){
                        return true;
                    }

            }
            if(baseRot2_[8] < 0.5){
                return true;
            }

            terminalReward = float(0.0);
            return false;
        }


        void updateObservation(const Eigen::Vector3d &command) {
            obDouble_.setZero(obDim_);

            /// body orientation - 3 + 3
            obDouble_.segment(0, 3) = baseRot_.e().row(2);
            obDouble_.segment(3, 3) = baseRot2_.e().row(2);
            /// body ang vel - 3 + 3
            obDouble_.segment(6, 3) = bodyAngVel_;
            obDouble_.segment(9, 3) = bodyAngVel2_;
            /// gc, gv - 3 + 3
            obDouble_.segment(12, 3) << gc_.tail(nJoints_);
            obDouble_.segment(15, 3) << gv_.tail(nJoints_);
            /// previous action - 3 + 3
            obDouble_.segment(18 , 3) = previousAction_;
            obDouble_.segment(21, 3) = prevprevAction_;

            /// command - 3
            obDouble_.segment(24, 3) << command;

        }

        void updateValueObservation(const Eigen::Vector3d &command) {
            valueObDouble_.setZero(valueObDim_);
            /// body orientation - 3
            /// body orientation - 3 + 3
            valueObDouble_.segment(0, 3) = baseRot_.e().row(2);
            valueObDouble_.segment(3, 3) = baseRot2_.e().row(2);
            /// body ang vel - 3 + 3
            valueObDouble_.segment(6, 3) = bodyAngVel_;
            valueObDouble_.segment(9, 3) = bodyAngVel2_;
            /// gc, gv - 3 + 3
            valueObDouble_.segment(12, 3) << gc_.tail(nJoints_);
            valueObDouble_.segment(15, 3) << gv_.tail(nJoints_);
            /// previous action - 3 + 3
            valueObDouble_.segment(18 , 3) = previousAction_;
            valueObDouble_.segment(21, 3) = prevprevAction_;
            /// command - 3
            valueObDouble_.segment(24, 3) << command;

            /// body lin vel - 3 + 3
            valueObDouble_.segment(27, 3) = bodyLinVel_;
            valueObDouble_.segment(30, 3) = bodyLinVel2_;
            /// height of the origin of the body frame - 1
            valueObDouble_[33] = baseHeight_;
        }

        inline void setRewardConfig(const Yaml::Node &cfg) {
            READ_YAML(double, commandTrackingRewardCoeff_, cfg["reward"]["command_tracking_reward"])
            READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
            READ_YAML(double, jointVelocityRewardCoeff_, cfg["reward"]["joint_velocity_reward_coeff"])
            READ_YAML(double, smoothRewardCoeff_, cfg["reward"]["smooth_reward_coeff"])
            READ_YAML(double, smoothReward2Coeff_, cfg["reward"]["smooth_reward2_coeff"])
            READ_YAML(double, orientationRewardCoeff_, cfg["reward"]["orientation_reward_coeff"])
            READ_YAML(double, conRewardCoeff_, cfg["reward"]["con_reward_coeff"])
            READ_YAML(double, basemotionRewardCoeff_, cfg["reward"]["base_motion_reward_coeff"])
            READ_YAML(double, clearanceRewardCoeff_, cfg["reward"]["clearance_reward_coeff"])
            READ_YAML(double, baseheightLimitRewardCoeff_, cfg["reward"]["base_height_limit_reward_coeff"])
            READ_YAML(double, airtimeRewardCoeff_, cfg["reward"]["airtime_reward_coeff"])
            READ_YAML(double, orientationReward2Coeff_, cfg["reward"]["orientation_reward2_coeff"])
        }

        inline void accumulateRewards(Eigen::Vector3d &command, double cf) {
            //commandTracking
            double linearCommandTrackingReward = 0., angularCommandTrackingReward = 0.;
            linearCommandTrackingReward += std::exp(-1.0 * (command.head(2) - bodyLinVel2_.head(2)).squaredNorm());
            angularCommandTrackingReward += std::exp(-1.0 * pow((command(2) - bodyAngVel2_(2)), 2));
//            if (command.head(2).norm() > 1.5)
//                linearCommandTrackingReward *= (1.0 + 0.5 * std::pow(command.head(2).norm() - 1.5, 2));
            commandTrackingReward_ += (1.5*linearCommandTrackingReward + 0.5*angularCommandTrackingReward) * commandTrackingRewardCoeff_;

            //torqueReward
            torqueReward_ += torqueRewardCoeff_ * (pogo_->getGeneralizedForce().e().tail(nJoints_).squaredNorm());

            //orientationReward
            orientationReward_ += orientationRewardCoeff_ * std::pow(baseRot_[8]+1,4);
            orientationReward2_ += orientationReward2Coeff_ * std::pow(baseRot2_[8]+1,4);

            //conReward
            if (footContactState_){
                conReward_ += conRewardCoeff_;
            }

            //smooth reward
            smoothReward_ += smoothRewardCoeff_ * (prepreprevAction_ + previousAction_ - 2 * prevprevAction_).cwiseProduct(smoothingWeight_).norm();
            smoothReward2_ += smoothReward2Coeff_ * (previousAction_ - prevprevAction_).cwiseProduct(smoothingWeight_).norm();

            //jointVelocityReward
            jointVelocityReward_ += jointVelocityRewardCoeff_ * jointVelocity_.norm();

            // basemotionReward
            basemotionReward_ += basemotionRewardCoeff_ * bodyAngVel_.head(2).norm();

            //clearanceReward
            if (baseHeight_ < desiredbaseHeight_) {
                clearanceReward_ += clearanceRewardCoeff_ * std::pow(baseHeight_ - desiredbaseHeight_, 2);
            } else {
                clearanceReward_ += clearanceRewardCoeff_ * 10.0 * std::pow(baseHeight_ - desiredbaseHeight_, 2);
            }


            if(baseHeight_ > 1.75){
                baseheightLimitReward_ += baseheightLimitRewardCoeff_ * pow(baseHeight_, 2);
            }


            //zvelReward
            if(airTime_ < 0.8){
                airtimeReward_ += airtimeRewardCoeff_ * airTime_;
            }



//            if (standingMode_) {
//                orientationReward *= 10.0;
//                basemotionReward *= 5.0;
//            }
        }

        void getLoggingInfo(const Eigen::Vector3d &command, Eigen::Ref<EigenVec> info) {
            Eigen::VectorXd infoBag;
            infoBag.setZero(24);


            infoBag << pogo_->getGeneralizedCoordinate().e().tail(nJoints_), // 0,1,2
                    pogo_->getGeneralizedVelocity().e().tail(nJoints_), // 3,4,5
                    pogo_->getGeneralizedForce().e().tail(nJoints_), // 6,7,8
                    bodyLinVel_, // 9,10,11
                    bodyAngVel_, // 12,13,14
                    command, // 15,16,17
                    pTarget_.tail(nJoints_), //18,19,20
                    airTime_, stanceTime_, //21,22
                    baseHeight_; //23

            info = infoBag.cast<float>();
        }

        Eigen::Vector3d getArrowPosition(){
            return gc_.head(3);
        }

        Eigen::Vector4d getCommandArrowOrientation(Eigen::Vector3d &command){
            double a = command[0];
            double b = command[1];
            double theta = std::atan2(b, a);

            Eigen::Vector4d quat;
            quat[0] = std::cos(theta / 2);
            quat[1] = 0.0;
            quat[2] = 0.0;
            quat[3] = std::sin(theta / 2);
            return quat;
        }

        Eigen::Vector4d getgvArrowOrientation(){
            double a = bodyLinVel_[0];
            double b = bodyLinVel_[1];
            double theta = std::atan2(b, a);

            Eigen::Vector4d quat;
            quat[0] = std::cos(theta / 2);
            quat[1] = 0.0;
            quat[2] = 0.0;
            quat[3] = std::sin(theta / 2);
            return quat;
        }


        inline void setStandingMode(bool mode) { standingMode_ = mode; }

        [[nodiscard]] static constexpr int getObDim() { return obDim_; }
        [[nodiscard]] static constexpr int getValueObDim() { return valueObDim_; }
        [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
        [[nodiscard]] static constexpr double getSimDt() { return simDt_; }
        [[nodiscard]] static constexpr double getConDt() { return conDt_; }
        void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

        void setSimDt(double dt) {};
        void setConDt(double dt) {};

        [[nodiscard]] inline const std::vector<std::string> &getStepDataTag() const { return stepDataTag_; }
        [[nodiscard]] inline const Eigen::VectorXd &getStepData() const { return stepData_; }

        // robot configuration variables
        raisim::ArticulatedSystem *pogo_;
        std::vector<size_t> footIndex_, baseIndex_;
        static constexpr int nJoints_ = 3;
        static constexpr int actionDim_ = 3;
        static constexpr size_t obDim_ = 27;
        static constexpr size_t valueObDim_ = 34;
        static constexpr double simDt_ = .0025;
        static constexpr double conDt_ = .01;
        int gcDim_ = 0;
        int gvDim_ = 0;

        // robot state variables
        Eigen::VectorXd gc_, gv_;
        Eigen::Vector3d bodyLinVel_, bodyLinVel2_, bodyAngVel_, bodyAngVel2_; /// body velocities are expressed in the body frame
        Eigen::VectorXd jointVelocity_;
        std::array<raisim::Vec<3>, 2> footPos_, relativeFootPos_, footVel_;
        raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
        bool footContactState_ = false;
        raisim::Mat<3, 3> baseRot_, baseRot2_, controlRot_;
        double airTime_, stanceTime_;
        double baseHeight_ = 0.0, desiredbaseHeight_ = 1.6;

        // pogo variables
        static constexpr size_t POGO_GC_PASSIVE_IDX = 0; // the gc index of the "passive" joint (needs pd gain set to spring const.)
        static constexpr size_t POGO_GC_PRISMATIC_IDX = 3; // the gc index of the "active" prismatic joint (needs different pd gain)
        static constexpr double pGainRev_ = 50.0;
        static constexpr double pGainPrsm_ = 5000.0;
        static constexpr double pogoSpringConstant_ = 15000.0;
        static constexpr double pogoPreload_ = 1000.0;


        // robot observation variables
        Eigen::VectorXd obDouble_;
        Eigen::VectorXd valueObDouble_;
        std::array<double, 2> contactNormalAngle_;

        // control variables
        bool standingMode_ = false;
        Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_, prevprevAction_, prepreprevAction_;
        Eigen::VectorXd actionDoulbe_;
        Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
        Eigen::VectorXd jointPgain_, jointDgain_;


        // reward variables
        double commandTrackingRewardCoeff_ = 0., commandTrackingReward_ = 0.;
        double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
        double smoothRewardCoeff_ = 0., smoothReward_ = 0.;
        double smoothReward2Coeff_ = 0., smoothReward2_ = 0.;
        double conRewardCoeff_ = 0., conReward_ = 0.;
        double jointVelocityRewardCoeff_ = 0., jointVelocityReward_ = 0.;
        double orientationRewardCoeff_ = 0., orientationReward_ = 0.;
        double orientationReward2Coeff_ = 0., orientationReward2_ = 0.;
        double basemotionRewardCoeff_ = 0., basemotionReward_ = 0.;
        double clearanceRewardCoeff_ = 0., clearanceReward_ = 0.;
        double baseheightLimitRewardCoeff_ = 0., baseheightLimitReward_ = 0.;
        double airtimeRewardCoeff_ = 0., airtimeReward_ = 0.;
        double terminalReward_ = -2000.0;
        Eigen::VectorXd smoothingWeight_;

        // exported data
        Eigen::VectorXd stepData_;
        std::vector<std::string> stepDataTag_;
    };

}

#endif //_RAISIM_GYM_pogo_CONTROLLER_HPP