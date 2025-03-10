#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <limits.h>
#include <cmath>
#include <chrono>

using namespace std;

void hide_features(vector <int> &current_set, vector <double> &object_to_classify, int feature_to_add);
double euclidean_distance(const vector <double> &vec1, const vector <double> &vec2);
double leave_one_out_cross_validation(vector <vector<double> > &data, int numColumns, int numRows, vector <int> &current_set, int feature_to_add);
double feature_search(vector <vector<double> > &data, int numColumns, int numRows);

int main(){

    string fileName;
    cout << "Type in the name of the file to test: ";
    cin >> fileName;

    cout << "Type in the number of the algorithm you want to run." << endl;
    cout << "1) Forward Selection\n2) Backward Elimination" << endl << endl;

    int algChoice;
    cin >> algChoice;
    
    //read data from chosen file
    ifstream file(fileName);
    if (!file) {
        cout << "Error opening " << fileName << endl;
        return 1;
    }

    vector <vector<double> > data;
    string line;
    int numColumns = 0;
    int numRows = 0;

    while (getline(file, line)) {
        istringstream iss(line);
        vector<double> row;
        double value;

        while (iss >> value) {
            row.push_back(value);
            numColumns++;
        }
        data.push_back(row);
        numRows++;
    }

    numColumns = (numColumns/numRows); //divide by the number of rows because numColumns is incremented at every row
    file.close();

    cout << "\nThis dataset has " << numColumns - 1 << " features (not including the class attribute), with " << numRows << " instances." << endl << endl;
    cout << "Running nearest neighbor with all " << numColumns - 1<< " features, using \"leave-one-out\" evaluation, I get an accuracy of %" << endl << endl;

    cout << "Beginning search." << endl;

    feature_search(data, numColumns, numRows);

}

double euclidean_distance(const vector <double> &vec1, const vector <double> &vec2){
    if (vec1.size() != vec2.size()){
        cout << "vector lengths are not equal" << endl;
        return -1.0;
    }

    double sum = 0.0;
    for (int i = 1; i < vec1.size(); ++i){ //skip class label, start at i = 1
        sum += pow(vec2[i] - vec1[i], 2);
    }
  
    return sqrt(sum);
}

void hide_features(vector <int> &current_set, vector <double> &object_to_classify, int feature_to_add){

    // cout << "BEFORE PRUNING object_to_classify: ";
    //     for (int n = 1; n < object_to_classify.size(); ++n){
    //         cout << object_to_classify[n] << " ";
    //     }
    // cout << endl;

    for (int i = 1; i < object_to_classify.size(); ++i){
        if (find(current_set.begin(), current_set.end(), i) == current_set.end() && i != feature_to_add){
            object_to_classify.at(i) = 0;
        }
    }

    // cout << "AFTER PRUNING object_to_classify:  ";
    //     for (int n = 1; n < object_to_classify.size(); ++n){
    //         cout << object_to_classify[n] << " ";
    //     }
    // cout << endl;
}

//function stub returns random accuracy to test feature_search
double leave_one_out_cross_validation(vector <vector<double> > &data, int numColumns, int numRows, vector <int> &current_set, int feature_to_add){
    double number_correctly_classified = 0.0;

    for (int i = 0; i < numRows - 1; ++i){
        vector <double> object_to_classify = data[i];
        double label_object_to_classify = object_to_classify[0];
        double nearest_neighbor_distance = INT_MAX;
        int nearest_neighbor_location = INT_MAX;
        double nearest_neighbor_label = -1;

        //modifies object_to_classify by hiding features not in current_set + feature_to_add
        hide_features(current_set, object_to_classify, feature_to_add);

        for (int k = 1; k < numRows - 1; ++k){
            if (k != i){
                vector <double> possible_nearest_neighbor = data[k];
                double distance = euclidean_distance(object_to_classify, possible_nearest_neighbor);
                //cout << "Distance = " << distance << endl;

                if (distance < nearest_neighbor_distance){
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k;
                    nearest_neighbor_label = data[nearest_neighbor_location - 1][0];
                }
            }
        }

        //cout << "Object " << i + 1 << " is in class " <<  label_object_to_classify << endl;
        //cout << "Its nearest neighbor is " << nearest_neighbor_location << " which is in class " <<  nearest_neighbor_label << endl;
        if (label_object_to_classify == nearest_neighbor_label){
            number_correctly_classified++;
        }
    }

    double accuracy = (number_correctly_classified/(static_cast<double> (numRows))) * 100;
    return accuracy;
}

double feature_search (vector <vector<double> > &data, int numColumns, int numRows){
    vector <int> current_features;
    int accuracy = 0;
    int best_accuracy = 0;

    for (int i = 1; i < numColumns; ++i){
        cout << "On the " << i << "th level of the search tree" << endl;
        int feature_to_add_at_this_level = 0; //CHANGED 0 to 1
        int best_accuracy_so_far = 0;

        for (int j = 1; j < numColumns; ++j){
            if (find(current_features.begin(), current_features.end(), j) == current_features.end()){
                cout << "--Considering adding the " << j << " feature" << endl;
                accuracy = leave_one_out_cross_validation (data, numColumns, numRows, current_features, j);
                cout << "Accuracy = " << accuracy << "%" << endl;
                if (accuracy > best_accuracy_so_far){
                    best_accuracy_so_far = accuracy;
                    feature_to_add_at_this_level = j;
                }
            }
        }

        best_accuracy = max(best_accuracy, best_accuracy_so_far);
        current_features.push_back(feature_to_add_at_this_level);
        
        cout << "Added " << feature_to_add_at_this_level << " to current set" << endl;

    }

    return best_accuracy;
}

//CS170_Small_Data__87.txt
//CS170_Large_Data__123.txt