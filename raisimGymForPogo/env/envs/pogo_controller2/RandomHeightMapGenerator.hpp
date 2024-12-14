//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_
#define _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_

#include "raisim/World.hpp"

namespace raisim {

    class RandomHeightMapGenerator {
    public:

        enum class GroundType : int {
            UNEVEN = 0,
            WEAK_UNEVEN = 1,
            STEPS = 2,
            SUDDEN_STEPS = 3,
            FLAT = 4,
            STAIRS = 5,
            SLOPE = 6,
        };

        RandomHeightMapGenerator() = default;

        void setSeed(int seed) {
            terrain_seed_ = seed;
        }

        raisim::HeightMap* generateTerrain(raisim::World* world,
                                           GroundType groundType,
                                           double curriculumFactor,
                                           std::mt19937& gen,
                                           std::uniform_real_distribution<double>& uniDist) {
            std::vector<double> heightVec;
            heightVec.resize(120*120);
            std::unique_ptr<raisim::TerrainGenerator> genPtr;
            const double targetRoughness = 0.5;
            const double maxAngle = 10. * (M_PI / 180.);
            const double slopeTargetAngle = 10. * (M_PI / 180.);

            switch (groundType) {
                case GroundType::UNEVEN: {
                    size_ = 20.;
                    gridSize_ = 0.1 + 0.1 * uniDist(gen);
                    nGrid_ = int(size_ / gridSize_);
                    terrainProperties_.frequency = 0.8;
                    terrainProperties_.zScale = targetRoughness * curriculumFactor;
                    terrainProperties_.xSize = size_;
                    terrainProperties_.ySize = size_;
                    terrainProperties_.xSamples = nGrid_;
                    terrainProperties_.ySamples = nGrid_;
                    terrainProperties_.fractalOctaves = 5;
                    terrainProperties_.fractalLacunarity = 3.0;
                    terrainProperties_.fractalGain = 0.45;
                    terrainProperties_.seed = terrain_seed_;
                    terrain_seed_ += 500;
                    terrainProperties_.stepSize = 0.;
                    genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
                    heightVec = genPtr->generatePerlinFractalTerrain();

                    return world->addHeightMap(nGrid_, nGrid_, size_, size_, 0., 0., heightVec);
                }

                case GroundType::WEAK_UNEVEN: {
                    size_ = 40.;
                    gridSize_ = 0.4 + 0.4 * uniDist(gen);
                    nGrid_ = int(size_ / gridSize_);
                    terrainProperties_.frequency = 0.8;
                    terrainProperties_.zScale = 0.5 * targetRoughness * curriculumFactor;
                    terrainProperties_.xSize = size_;
                    terrainProperties_.ySize = size_;
                    terrainProperties_.xSamples = nGrid_;
                    terrainProperties_.ySamples = nGrid_;
                    terrainProperties_.fractalOctaves = 5;
                    terrainProperties_.fractalLacunarity = 3.0;
                    terrainProperties_.fractalGain = 0.45;
                    terrainProperties_.seed = terrain_seed_;
                    terrain_seed_ += 500;
                    terrainProperties_.stepSize = 0.;
                    genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
                    heightVec = genPtr->generatePerlinFractalTerrain();

                    return world->addHeightMap(nGrid_, nGrid_, size_, size_, 0., 0., heightVec);
                }

                case GroundType::STEPS: {
                    width_ = 0.1 + 0.4 * abs(uniDist(gen));
                    height_ = 0.2 * curriculumFactor;

                    size_ = 20.;
                    gridSize_ = 0.04;
                    nGrid_ = int(size_ / gridSize_);
                    nBlock_ = int(size_ / width_);

                    nGridPerBlock_ = int(nGrid_ / nBlock_);
                    nGrid_ = nGridPerBlock_ * nBlock_;
                    size_ = nGrid_ * gridSize_;

                    heightVec.resize(nGrid_ * nGrid_);
                    for (int xBlock = 0; xBlock < nBlock_; ++xBlock) {
                        for (int yBlock = 0; yBlock < nBlock_; ++yBlock) {
                            double stepHeight = height_ * uniDist(gen);
                            for (int i = 0; i < nGridPerBlock_; i++) {
                                for (int j = 0; j < nGridPerBlock_; ++j) {
                                    heightVec[nGrid_ * (nGridPerBlock_ * yBlock + j) + (nGridPerBlock_ * xBlock + i)] = stepHeight;
                                }
                            }
                        }
                    }

                    return world->addHeightMap(nGrid_, nGrid_, size_, size_, 0., 0., heightVec);
                }

                case GroundType::SUDDEN_STEPS: {
                    width_ = 1.0;
                    height_ = 0.2 * curriculumFactor;

                    size_ = 20.;
                    gridSize_ = 0.025;
                    nGrid_ = int(size_ / gridSize_);
                    nBlock_ = int(size_ / width_);

                    nGridPerBlock_ = int(nGrid_ / nBlock_);
                    nGrid_ = nGridPerBlock_ * nBlock_;
                    size_ = nGrid_ * gridSize_;
                    width_ = size_ / double(nBlock_);

                    heightVec.resize(nGrid_ * nGrid_);


                    for (int xBlock = 0; xBlock < nBlock_; ++xBlock) {
                        for (int yBlock = 0; yBlock < nBlock_; ++yBlock) {
                            double stepHeight = 0.;
                            if (uniDist(gen) > 0.5) {
                                stepHeight = height_ * (0.7 + 0.3 * uniDist(gen));
                            }

                            for (int i = 0; i < nGridPerBlock_; ++i) {
                                for (int j = 0; j < nGridPerBlock_; ++j) {
                                    heightVec[nGrid_ * (nGridPerBlock_ * yBlock + j) + (nGridPerBlock_ * xBlock + i)] = stepHeight;
                                }
                            }
                        }
                    }

                    return world->addHeightMap(nGrid_, nGrid_, size_, size_, 0., 0., heightVec);

                }

                case GroundType::FLAT:
                    heightVec.resize(200 * 200, 0.5);
                    return world->addHeightMap(200, 200, 12, 12, 0., 0., heightVec);

                case GroundType::STAIRS: {
                    width_ = 0.1 + 0.4 * uniDist(gen);
                    double maxHeight = 0.22;

                    double p = uniDist(gen);
                    if (p < 0.25) height_ = std::abs(maxHeight * curriculumFactor);
                    else height_ = std::abs(maxHeight * curriculumFactor * (1. + uniDist(gen)) / 2.);

                    width_ = std::max(width_, height_ / std::tan(maxAngle));

                    size_ = 20.;
                    gridSize_ = 0.025;
                    nGrid_ = int(size_ / gridSize_);
                    nBlock_ = int(size_ / width_);

                    nGridPerBlock_ = int(nGrid_ / nBlock_);
                    nGrid_ = nGridPerBlock_ * nBlock_;
                    size_ = nGrid_ * gridSize_;
                    width_ = size_ / double(nBlock_);

                    heightVec.resize(4 * nGrid_);



                    for (int yBlock = 0; yBlock < nBlock_; ++yBlock) {
                        for (int i = 0; i < 4 * nGridPerBlock_; ++i) {
                            heightVec[4 * nGridPerBlock_ * yBlock + i] = yBlock * height_;
                        }
                    }
                    return world->addHeightMap(4, nGrid_, size_, size_, 0., 0., heightVec);
                }


                case GroundType::SLOPE: {
                    size_ = 20.;
                    gridSize_ = 0.1;
                    nGrid_ = int(size_ / gridSize_);
                    terrainProperties_.frequency = 0.8;
                    terrainProperties_.zScale = targetRoughness * curriculumFactor;
                    terrainProperties_.xSize = size_;
                    terrainProperties_.ySize = size_;
                    terrainProperties_.xSamples = nGrid_;
                    terrainProperties_.ySamples = nGrid_;
                    terrainProperties_.fractalOctaves = 5;
                    terrainProperties_.fractalLacunarity = 3.0;
                    terrainProperties_.fractalGain = 0.45;
                    terrainProperties_.seed = terrain_seed_;
                    terrain_seed_ += 500;
                    terrainProperties_.stepSize = 0.;
                    genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
                    heightVec = genPtr->generatePerlinFractalTerrain();

                    std::vector<double> heightVecTmp(heightVec.size(), 0.);

                    double angle;
                    for (int yBlock = 1; yBlock < terrainProperties_.ySamples; ++yBlock) {
//          if (uniDist(gen) < 0.05) angle = slopeWorstAngle;
//          else angle = slopeTargetAngle;
                        angle = slopeTargetAngle;
                        for (int i = 0; i < terrainProperties_.xSamples; ++i) {
                            heightVecTmp[yBlock * terrainProperties_.xSamples + i] = heightVecTmp[(yBlock - 1) * terrainProperties_.xSamples + i] +
                                                                                     tan(angle)*(terrainProperties_.ySize/terrainProperties_.ySamples) * curriculumFactor;
                            heightVec[yBlock * terrainProperties_.xSamples + i] += heightVecTmp[yBlock * terrainProperties_.xSamples + i];
                        }
                    }

                    return world->addHeightMap(terrainProperties_.xSamples, terrainProperties_.ySamples, terrainProperties_.xSize, terrainProperties_.ySize, 0., 0., heightVec);
                }
            }
            return nullptr;
        }

    private:
        raisim::TerrainProperties terrainProperties_;
        int terrain_seed_;
        int nGrid_, nBlock_, nGridPerBlock_;
        double gridSize_, size_, width_, height_;
    };

}

#endif //_RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_