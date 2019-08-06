#include <ORBextractor.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    cv::Mat imgLeft_ = cv::imread(argv[1], 0);
    cv::Mat imgRight_ = cv::imread(argv[2], 0);
    cv::Mat imgLeft_d = cv::imread(argv[3], 0);
    cv::Mat imgRight_d = cv::imread(argv[4], 0);

    cv::Mat imgLeft_c = cv::imread(argv[1], -1);
    cv::Mat imgRight_c = cv::imread(argv[2], -1);

    if (imgLeft_.empty() || imgRight_.empty() || imgLeft_d.empty() || imgRight_d.empty())
        return -1;

    cv::Mat descLeft_, descRight_;
    vector<cv::KeyPoint> kpsLeft_, kpsRight_;
    std::vector<cv::DMatch> matches_1st, matches_2nd, matches_;

    ORB_SLAM2::ORBextractor* extractorL = new ORB_SLAM2::ORBextractor(2000, 1.2, 8, 20, 7);
    ORB_SLAM2::ORBextractor* extractorR = new ORB_SLAM2::ORBextractor(2000, 1.2, 8, 20, 7);
    (*extractorL)(imgLeft_, cv::Mat(), kpsLeft_, descLeft_);
    (*extractorR)(imgRight_, cv::Mat(), kpsRight_, descRight_);

    // erase keypoints without depth
    int bad_key_point_l = 0, bad_key_point_r = 0;
    vector<cv::KeyPoint> kpstmp;
    cv::Mat desctmp;
    for (size_t i=0; i<kpsLeft_.size(); i++) {
        float d = imgLeft_d.at<uchar>(kpsLeft_[i].pt.y, kpsLeft_[i].pt.x);
        if (d>0 && d<=6.5) {
            kpstmp.push_back(kpsLeft_[i]);
            desctmp.push_back(descLeft_.row(i).clone());
        } else {
            bad_key_point_l++;
        }
    }
    kpsLeft_.swap(kpstmp);
    descLeft_ = desctmp.clone();
    kpstmp.clear();
    desctmp.release();

    for (size_t i=0; i<kpsRight_.size(); i++) {
        float d = imgRight_d.at<uchar>(kpsRight_[i].pt.y, kpsRight_[i].pt.x);
        if (d>0 && d<=6.5) {
            kpstmp.push_back(kpsRight_[i]);
            desctmp.push_back(descRight_.row(i).clone());
        } else {
            bad_key_point_r++;
        }
    }
    kpsRight_.swap(kpstmp);
    descRight_ = desctmp.clone();
    cout << "bad_key_point: " << bad_key_point_l << ", " << bad_key_point_r << endl;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descLeft_, descRight_, matches_1st);

    // 找到正确的匹配
    double min_dist = 10000;
    for (auto& m : matches_1st ) {
        double dist = m.distance;
        if (dist < min_dist) { min_dist = dist; }
    }
    for (auto& m : matches_1st ) {
        if ( m.distance <= std::max(2 * min_dist, 60.0) ) {
            matches_2nd.push_back(m);
        }
    }
    cv::Mat image_show;
    cv::drawMatches(imgLeft_, kpsLeft_, imgRight_, kpsRight_, matches_2nd, image_show);
    cv::imshow("matched before ransac", image_show);
    cv::imwrite("before_ransac.png", image_show);

    // RANSAC 再次消除误匹配
    vector<KeyPoint> kpsLeft_2nd, kpsRight_2nd;
    vector<Point2f> kpl, kpr;
    Mat descLeft_2nd;
    for (size_t i=0; i<matches_2nd.size(); i++) {
       // 经过此步 kpsLeft_2nd 和 kpsRight_2nd 在顺序上有一一匹配关系了
       kpsLeft_2nd.push_back(kpsLeft_[matches_2nd[i].queryIdx]);
       kpsRight_2nd.push_back(kpsRight_[matches_2nd[i].trainIdx]);
       kpl.push_back(kpsLeft_2nd[i].pt);
       kpr.push_back(kpsRight_2nd[i].pt);
       descLeft_2nd.push_back(descLeft_.row(matches_2nd[i].queryIdx).clone());
    }

    vector<uchar> RansacStatus;
    Mat H = findHomography(kpl, kpr, CV_RANSAC, 30, RansacStatus);

    vector<KeyPoint> aft_ran_kpsl, aft_ran_kpsr;
    Mat aft_ran_descl;
    int index = 0;
    for (size_t i=0; i<matches_2nd.size(); i++) {
       if (RansacStatus[i] != 0) {
           aft_ran_kpsl.push_back(kpsLeft_2nd[i]);
           aft_ran_kpsr.push_back(kpsRight_2nd[i]);
           matches_2nd[i].queryIdx = index;
           matches_2nd[i].trainIdx = index;
           matches_.push_back(matches_2nd[i]);  // 匹配的索引基于aft_ran_kps
           aft_ran_descl.push_back(descLeft_2nd.row(i).clone());
           index++;
       }
    }
    cout << "matched: " << matches_.size() << endl;
    // 更新特征点和对应的描述子成员变量
    kpsLeft_.swap(aft_ran_kpsl);
    kpsRight_.swap(aft_ran_kpsr);
    descLeft_ = aft_ran_descl.clone();

    cv::drawMatches(imgLeft_, kpsLeft_, imgRight_, kpsRight_, matches_, image_show);
    cv::imshow("matched after ransac", image_show);
    cv::imwrite("after_ransac.png", image_show);
    cv::waitKey(0);

    return 0;
}
