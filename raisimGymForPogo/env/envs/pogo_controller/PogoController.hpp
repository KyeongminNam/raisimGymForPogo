//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_segway_CONTROLLER_HPP
#define _RAISIM_GYM_segway_CONTROLLER_HPP

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

            /// exported data
            stepDataTag_ = {"command_tracking_rew",
                            "torque_rew",
                            "joint_vel_rew",
                            "smooth_rew",
                            "smooth_rew2",
                            "orientation_rew",
                            "con_rew",
                            "base_motion_rew",
                            "base_height_rew",
                            "positive_rew",
                            "negative_rew"};
            stepData_.resize(stepDataTag_.size());

            return true;
        };

        void updateStateVariables(const raisim::HeightMap *map, std::mt19937 &gen_,
                                  std::normal_distribution<double> &normDist_) {
            pogo_->getState(gc_, gv_);
            jointVelocity_ = gv_.tail(nJoints_);
            baseRot_ = pogo_->getBaseOrientation();
            bodyLinVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
            bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);
            baseHeight_ = gc_[2];


            /// airtime & standtime
            for (int i = 0; i < 2; i++) {
                if (footContactState_[i]) {
                    airTime_[i] = 0;
                    stanceTime_[i] += simDt_;
                } else {
                    airTime_[i] += simDt_;
                    stanceTime_[i] = 0;
                }
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

        [[nodiscard]] float getRewardSum(const int & howManySteps) {
            double positiveReward, negativeReward, totalReward;
            stepData_[0] = commandTrackingReward_;
            stepData_[1] = torqueReward_;
            stepData_[2] = jointVelocityReward_;
            stepData_[3] = smoothReward_;
            stepData_[4] = smoothReward2_;
            stepData_[5] = orientationReward_;
            stepData_[6] = conReward_;
            stepData_[7] = basemotionReward_;
            stepData_[8] = baseheightReward_;
            stepData_[9] = 0.0;
            stepData_[10] = 0.0;
            stepData_ /= howManySteps;
            totalReward = stepData_.sum();
            positiveReward = stepData_[0] + stepData_[5] + stepData_[8];
            negativeReward = totalReward - positiveReward;
            stepData_[9] = positiveReward;
            stepData_[10] = negativeReward;

            commandTrackingReward_ = 0.;
            torqueReward_ = 0.;
            jointVelocityReward_ = 0.;
            smoothReward_ = 0.;
            smoothReward2_ = 0.;
            orientationReward_ = 0.;
            conReward_ = 0.;
            basemotionReward_ = 0.;
            baseheightReward_ = 0.;

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

            terminalReward = float(0.0);
            return false;
        }


        void updateObservation(const Eigen::Vector3d &command) {
            obDouble_.setZero(obDim_);

            /// body orientation - 3
            obDouble_.segment(0, 3) = baseRot_.e().row(2);
            /// body ang vel - 3
            obDouble_.segment(3, 3) = bodyAngVel_;
            /// except the first joints, the joint history stores target-position - 3
            obDouble_.segment(6, 3) << gc_.tail(nJoints_);
            /// previous action - 6
            obDouble_.segment(9 , 3) = previousAction_;
            obDouble_.segment(12, 3) = prevprevAction_;
            /// command - 3
            obDouble_.segment(15, 3) << command;

        }

        void updateValueObservation(const Eigen::Vector3d &command) {
            valueObDouble_.setZero(valueObDim_);
            /// body orientation - 3
            valueObDouble_.segment(0, 3) = baseRot_.e().row(2);
            /// body ang vel - 3
            valueObDouble_.segment(3, 3) = bodyAngVel_;
            /// except the first joints, the joint history stores target-position - 3
            valueObDouble_.segment(6, 3) << gc_.tail(nJoints_);
            /// previous action - 6
            valueObDouble_.segment(9 , 3) = previousAction_;
            valueObDouble_.segment(12, 3) = prevprevAction_;
            /// command - 3
            valueObDouble_.segment(15, 3) << command;

            /// body lin vel - 3
            valueObDouble_.segment(18, 3) = bodyLinVel_;
            /// height of the origin of the body frame - 1
            valueObDouble_[21] = baseHeight_;

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
            READ_YAML(double, baseheightRewardCoeff_, cfg["reward"]["base_height_reward_coeff"])
        }

        inline void accumulateRewards(Eigen::Vector3d &command, double cf) {
            //commandTracking
            double linearCommandTrackingReward = 0., angularCommandTrackingReward = 0.;
            linearCommandTrackingReward += std::exp(-1.0 * pow((command(0) - bodyLinVel_(0)), 2));
            angularCommandTrackingReward += std::exp(-1.5 * pow((command(2) - bodyAngVel_(2)), 2));
            if (command(0) > 1.5)
                linearCommandTrackingReward *= std::min((1.0 + 0.5 * std::pow(command(0) - 1.5, 2)), 5.0);
            commandTrackingReward_ += (linearCommandTrackingReward + angularCommandTrackingReward) * commandTrackingRewardCoeff_;

            //torqueReward
            torqueReward_ += torqueRewardCoeff_ * (pogo_->getGeneralizedForce().e().tail(6).squaredNorm());

            //orientationReward
            orientationReward_ = orientationRewardCoeff_ * baseRot_[8];

            //conReward
            for (int i=0; i<2; i++)
                if (!footContactState_[i]){
                    conReward_ += conRewardCoeff_;
                }

            //smooth reward
            smoothReward_ += smoothRewardCoeff_ * (prepreprevAction_ + previousAction_ - 2 * prevprevAction_).norm();
            smoothReward2_ += smoothReward2Coeff_ * (previousAction_ - prevprevAction_).norm();

            //jointVelocityReward
            jointVelocityReward_ = jointVelocityRewardCoeff_ * jointVelocity_.norm();

            // basemotionReward
            basemotionReward_ = basemotionRewardCoeff_ * bodyAngVel_.head(2).squaredNorm();

            //baseheightReward
            baseheightReward_ = baseheightRewardCoeff_ * std::min(baseHeight_, 4.0);


//            if (standingMode_) {
//                orientationReward *= 10.0;
//                basemotionReward *= 5.0;
//            }
        }

        void getLoggingInfo(const Eigen::Vector3d &command, Eigen::Ref<EigenVec> info) {
        }


        inline void setStandingMode(bool mode) { standingMode_ = mode; }

        [[nodiscard]] static constexpr int getObDim() { return obDim_; }
        [[nodiscard]] static constexpr int getValueObDim() { return valueObDim_; }
        [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
        [[nodiscard]] double getSimDt() { return simDt_; }
        [[nodiscard]] double getConDt() { return conDt_; }
        void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

        void setSimDt(double dt) { simDt_ = dt; };
        void setConDt(double dt) { conDt_ = dt; };

        [[nodiscard]] inline const std::vector<std::string> &getStepDataTag() const { return stepDataTag_; }
        [[nodiscard]] inline const Eigen::VectorXd &getStepData() const { return stepData_; }

        // robot configuration variables
        raisim::ArticulatedSystem *pogo_;
        static constexpr int nJoints_ = 3;
        static constexpr int actionDim_ = 3;
        static constexpr size_t obDim_ = 18;
        static constexpr size_t valueObDim_ = 22;
        double simDt_ = .0025;
        int gcDim_ = 0;
        int gvDim_ = 0;

        // robot state variables
        Eigen::VectorXd gc_, gv_;
        Eigen::Vector3d bodyLinVel_, bodyAngVel_; /// body velocities are expressed in the body frame
        Eigen::VectorXd jointVelocity_;
        std::array<raisim::Vec<3>, 2> footPos_, relativeFootPos_, footVel_;
        raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
        std::array<bool, 2> footContactState_;
        raisim::Mat<3, 3> baseRot_, controlRot_;
        Eigen::Vector2d airTime_, stanceTime_;
        double baseHeight_ = 0.0;

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
        double conDt_ = 0.01;
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
        double basemotionRewardCoeff_ = 0., basemotionReward_ = 0.;
        double baseheightRewardCoeff_ = 0., baseheightReward_ = 0.;
        double terminalReward_ = -100.0;

        // exported data
        Eigen::VectorXd stepData_;
        std::vector<std::string> stepDataTag_;
    };

}

#endif //_RAISIM_GYM_segway_CONTROLLER_HPP