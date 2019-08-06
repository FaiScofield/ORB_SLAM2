#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;
using boost::ends_with;
namespace fs = boost::filesystem;

struct OdomVI {
    long long int timestamp;
    double x, y, theta;

    OdomVI() { timestamp = 0; x = y = theta = 0.0; }
};

struct OdomRaw {
    long long int timestamp;
    double x, y, theta;
    double linearVelX, AngularVelZ;
    double deltaDistance, deltaTheta;

    OdomRaw() {
        timestamp = 0;
        x = y = theta = 0.0;
        linearVelX = AngularVelZ = 0.0;
        deltaDistance = deltaTheta = 0.0;
    }
};

struct OdomFrame {
    long long int timestamp;
    double linearVelX, AngularVelZ;
    double deltaDistance, deltaTheta;

    OdomFrame() {
        timestamp = 0;
        linearVelX = AngularVelZ = 0.0;
        deltaDistance = deltaTheta = 0.0;
    }
};


vector<string> readFolderFiles(const string& folder) {
    vector<string> fileNames;

    fs::path path(folder);
    if (!fs::exists(path)) {
        fprintf(stderr, "folder does not exist!\n");
        return fileNames;
    }

    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (fs::is_regular_file(iter->status()))
            fileNames.push_back(iter->path().string());
    }

    if (fileNames.empty())
        fprintf(stderr, "Not image data in the folder!\n");
    else
        printf("Read %ld images in the folder.\n", fileNames.size());

    sort(fileNames.begin(), fileNames.end());
    return fileNames;
}

vector<OdomFrame> readOdomeFrame(const string& file) {
    vector<OdomFrame> result;

    ifstream reader;
    reader.open(file.c_str());
    if (!reader) {
        fprintf(stderr, "%s file open error!\n", file.c_str());
        return result;
    }

    // get data
    while (reader.peek() != EOF) {
        OdomFrame ofr;
        reader >> ofr.timestamp >> ofr.deltaTheta >> ofr.AngularVelZ
               >> ofr.deltaDistance >> ofr.linearVelX;
        if (ofr.timestamp != 0)
            result.push_back(ofr);
    }

    reader.close();

    return result;
}

vector<OdomRaw> readOdomeRaw(const string& file) {
    vector<OdomRaw> result;

    ifstream reader;
    reader.open(file.c_str());
    if (!reader) {
        fprintf(stderr, "%s file open error!\n", file.c_str());
        return result;
    }

    // get data
    while (reader.peek() != EOF) {
        OdomRaw oraw;
        reader >> oraw.timestamp >> oraw.x >> oraw.y >> oraw.theta
               >> oraw.linearVelX >> oraw.AngularVelZ
               >> oraw.deltaDistance >> oraw.deltaTheta;
        if (oraw.timestamp != 0)
            result.push_back(oraw);
    }

    reader.close();

    return result;
}

vector<OdomVI> readOdomeVI(const string& file) {
    vector<OdomVI> result;

    ifstream reader;
    reader.open(file.c_str());
    if (!reader) {
        fprintf(stderr, "%s file open error!\n", file.c_str());
        return result;
    }

    // get data
    while (reader.peek() != EOF) {
        OdomVI ovi;
        reader >> ovi.timestamp >> ovi.x >> ovi.y >> ovi.theta;
        if (ovi.timestamp != 0)
            result.push_back(ovi);
    }

    reader.close();

    return result;
}

void dealImage(const Mat& image) {
    imshow("current image", image);
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_folder>\n", argv[0]);
        return -1;
    }

    string fileDeltaFrameOdom, fileOdomRaw, fileOdomVI, imagePath, suffix;
    string dataFolder = argv[1];
    if (ends_with(dataFolder, "/")) {
        suffix = "";
    } else {
        suffix = "/";
    }
    fileDeltaFrameOdom  = dataFolder + suffix + "deltaFrameOdom.txt";
    fileOdomRaw         = dataFolder + suffix + "odomRaw.txt";
    fileOdomVI          = dataFolder + suffix + "odomVI.txt";
    imagePath           = dataFolder + suffix + "slam_img/";

    vector<string> imageFiles = readFolderFiles(imagePath);
    vector<OdomFrame> odomFrame = readOdomeFrame(fileDeltaFrameOdom);
    vector<OdomRaw> odomRaw = readOdomeRaw(fileOdomRaw);
    vector<OdomVI> odomVI = readOdomeVI(fileOdomVI);

    for (size_t i = 0; i < imageFiles.size(); ++i) {
        Mat currentImage = imread(imageFiles[i], -1);
        dealImage(currentImage);
        waitKey(30);
    }

    return 0;
}
