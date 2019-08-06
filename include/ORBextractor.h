/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <list>
#include <opencv/cv.h>
#include <vector>


namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    ~ORBextractor() {}

    // Compute the ORB features and descriptors on an image.    计算图像的orb特征及描述
    // ORB are dispersed on the image using an octree.          将orb特征分配到一个四叉树当中
    // Mask is ignored in the current implementation.           目前mask参数是被忽略的，没有实现
    void operator()(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints,
                    cv::OutputArray descriptors);

    int inline GetLevels() { return nlevels; }

    float inline GetScaleFactor() { return scaleFactor; }

    std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

    std::vector<float> inline GetInverseScaleFactors() { return mvInvScaleFactor; }

    std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

    std::vector<float> inline GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,
                                                const int &minX, const int &maxX, const int &minY,
                                                const int &maxY, const int &nFeatures,
                                                const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;  //每层的特征数量

    std::vector<int> umax;  // Patch圆的最大坐标,用于存储计算特征方向时，图像每个v坐标对应最大的u坐标

    std::vector<float> mvScaleFactor;     //每层的相对于原始图像的缩放比例
    std::vector<float> mvInvScaleFactor;  //每层的相对于原始图像的缩放比例的倒数
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

}  // namespace ORB_SLAM

#endif
