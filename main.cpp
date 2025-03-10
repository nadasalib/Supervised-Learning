#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <limits.h>
#include <cmath>
#include <iomanip>
#include <ctime>

using namespace std;

void hide_features(vector <int> &current_set, int feature_to_add, vector <double> &object_to_prune);
double euclidean_distance(const vector <double> &vec1, const vector <double> &vec2);
double leave_one_out_cross_validation(vector <vector<double> > &data, int numColumns, int numRows, vector <int> &current_set, int feature_to_add);
void feature_search(vector <vector<double> > &data, int numColumns, int numRows);

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
    cout << "Beginning search." << endl;

    std::clock_t start = std::clock();
    feature_search(data, numColumns, numRows);
    std::clock_t end = std::clock();
    double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Total time = " << time << " seconds" << endl;
}

double euclidean_distance(const vector<double> &vec1, const vector<double> &vec2) {
    if (vec1.size() != vec2.size()) {
        cerr << "Error: Vector lengths are not equal" << endl;
        return -1.0;
    }

    double sum = 0.0;
    for (size_t i = 1; i < vec1.size(); ++i) {  // Skip first element
        double diff = vec2[i] - vec1[i];
        sum += diff * diff;
    }

    double result = sqrt(sum);
    return result;
}

void hide_features(vector <int> &current_set, int feature_to_add, vector <double> &object_to_prune){

    // cout << "BEFORE PRUNING object_to_prune: ";
    //     for (int n = 1; n < object_to_prune.size(); ++n){
    //         cout << object_to_prune[n] << " ";
    //     }
    // cout << endl;

    for (int i = 1; i < object_to_prune.size(); ++i){
        if (find(current_set.begin(), current_set.end(), i) == current_set.end() && i != feature_to_add){
            object_to_prune.at(i) = 0;
        }
    }

    // cout << "AFTER PRUNING object_to_prune:  ";
    //     for (int n = 1; n < object_to_prune.size(); ++n){
    //         cout << object_to_prune[n] << " ";
    //     }
    // cout << endl;
}

//function stub returns random accuracy to test feature_search
double leave_one_out_cross_validation(vector <vector<double> > &data, int numColumns, int numRows, vector <int> &current_set, int feature_to_add){
    double number_correctly_classified = 0;

    for (int i = 0; i < numRows - 1; ++i){
        vector <double> object_to_classify = data[i];
        double label_object_to_classify = object_to_classify[0];
        double nearest_neighbor_distance = INT_MAX;
        int nearest_neighbor_location = INT_MAX;
        double nearest_neighbor_label = -1;

        //modifies object_to_classify by hiding features not in current_set + feature_to_add
        hide_features(current_set, feature_to_add, object_to_classify);

        for (int k = 1; k < numRows - 1; ++k){
            if (k != i){
                vector <double> possible_nearest_neighbor = data[k];
                hide_features(current_set, feature_to_add, possible_nearest_neighbor);
                double distance = euclidean_distance(object_to_classify, possible_nearest_neighbor);

                if (distance < nearest_neighbor_distance){
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k;
                    nearest_neighbor_label = possible_nearest_neighbor[0];
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

void feature_search (vector <vector<double> > &data, int numColumns, int numRows){
    vector <int> current_features;
    double accuracy = 0;
    double best_accuracy = 0;
    vector <int> best_features;

    for (int i = 1; i < numColumns; ++i){
        cout << "On the " << i << "th level of the search tree" << endl;
        int feature_to_add_at_this_level = 0;
        double best_accuracy_so_far = 0;

        for (int j = 1; j < numColumns; ++j){
            if (find(current_features.begin(), current_features.end(), j) == current_features.end()){
                cout << "Considering feature " << j << endl;
                accuracy = leave_one_out_cross_validation (data, numColumns, numRows, current_features, j);
                cout << "\tUsing feature(s) {";
                for (int f = 0; f < current_features.size(); ++f){
                    cout << current_features.at(f) << ", ";
                } 
                cout << j << "} accuracy is " << setprecision(3) << accuracy << "%" << endl;
                
                if (accuracy > best_accuracy_so_far){
                    best_accuracy_so_far = accuracy;
                    feature_to_add_at_this_level = j;
                }
            }
        }

        best_accuracy = max(best_accuracy, best_accuracy_so_far);
        current_features.push_back(feature_to_add_at_this_level);

        if (best_accuracy == best_accuracy_so_far){
            best_features = current_features;
        }
        
        cout << "Feature set {";
            for (int f = 0; f < current_features.size() - 1; ++f){
                cout << current_features.at(f) << ", ";
            } 
        cout << current_features.back() << "} was best, accuracy is " << best_accuracy_so_far << "%" << endl;

    }

    //cout << "Running nearest neighbor with all " << numColumns - 1 << " features, using \"leave-one-out\" evaluation, I get an accuracy of " << best_accuracy << "%" << endl;
    cout << "Finished search! The best feature subset is {";
            for (int f = 0; f < best_features.size() - 1; ++f){
                cout << best_features.at(f) << ", ";
            } 
    cout << best_features.back() << "}, which has an accuracy of " << best_accuracy << "%" << endl;
}

//CS170_Small_Data__87.txt
//CS170_Large_Data__123.txt