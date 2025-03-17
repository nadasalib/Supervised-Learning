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

double default_rate(vector <vector<double> > &data, int numRows);
void hide_features(vector <int> &current_set, int feature, vector <double> &object_to_prune, bool directionFlag);
double euclidean_distance(const vector <double> &vec1, const vector <double> &vec2);
double leave_one_out_cross_validation(vector <vector<double> > &data, int numColumns, int numRows, vector <int> &current_set, int feature_to_add, bool directionFlag);
void forward_selection(vector <vector<double> > &data, int numColumns, int numRows);
void backward_elimination (vector <vector<double> > &data, int numColumns, int numRows);

int main(){

    string fileName;
    cout << "Type in the name of the file to test: ";
    cin >> fileName;

    cout << "Type in the number of the algorithm you want to run." << endl;
    cout << "1) Forward Selection\n2) Backward Elimination" << endl << endl;

    int algChoice;
    cin >> algChoice;

    while (algChoice != 1 && algChoice != 2){
        cout << "Invalid choice, please type 1 or 2" << endl;
        cin >> algChoice;
    }
    
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

    std::clock_t start = std::clock();
    if (algChoice == 1){
        forward_selection(data, numColumns, numRows);
    } else if (algChoice == 2){
        backward_elimination(data, numColumns, numRows);
    }
    std::clock_t end = std::clock();
    double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Total time = " << time << " seconds" << endl;
}

//finds accuracy using greatest occurence of one class label/total number of instances (no features)
double default_rate(vector <vector<double> > &data, int numRows){
    double numClass1 = 0;
    double numClass2 = 0;
    for (int i = 0; i < numRows; ++i){
        if (data[i][0] == 1){
            numClass1++;
        } else {
            numClass2++;
        }
    }
    return (max(numClass1, numClass2)/numRows) * 100;
}

//uses euclidean distance formula to find nearest neighbor
double euclidean_distance(const vector<double> &vec1, const vector<double> &vec2) {
    if (vec1.size() != vec2.size()) {
        cout << "Error: Vector lengths are not equal" << endl;
        return -1.0;
    }

    double sum = 0.0;
    // Skip class label when calculating distance
    for (size_t i = 1; i < vec1.size(); ++i) {
        double diff = vec2[i] - vec1[i];
        sum += diff * diff;
    }

    double result = sqrt(sum);
    return result;
}

//sets current object's unnecessary features to 0
void hide_features(vector <int> &current_set, int feature, vector <double> &object_to_prune, bool directionFlag){ //direction flag = 0 for forward, 1 for backward

    if (directionFlag == 0){ //forward selection
        for (int i = 1; i < object_to_prune.size(); ++i){
            if (find(current_set.begin(), current_set.end(), i) == current_set.end() && i != feature){
                object_to_prune.at(i) = 0;
            }
        }
    } else { //backward elimination 
        for (int i = 1; i < object_to_prune.size(); ++i){
            if (find(current_set.begin(), current_set.end(), i) == current_set.end() || i == feature){
                object_to_prune.at(i) = 0;
            }
        }
    }
}

//returns accuracy based on number of correctly classified nearest neighbors
double leave_one_out_cross_validation(vector <vector<double> > &data, int numColumns, int numRows, vector <int> &current_set, int feature, bool directionFlag){
    double number_correctly_classified = 0;

    for (int i = 0; i < numRows - 1; ++i){
        vector <double> object_to_classify = data[i];
        double label_object_to_classify = object_to_classify[0];
        double nearest_neighbor_distance = INT_MAX;
        int nearest_neighbor_location = INT_MAX;
        double nearest_neighbor_label = -1;

        //modifies object_to_classify by hiding features not in current_set + feature_to_add
        hide_features(current_set, feature, object_to_classify, directionFlag);

        //calculate the distance of the current object with every other object to find nearest neighbor
        for (int k = 1; k < numRows - 1; ++k){
            if (k != i){
                vector <double> possible_nearest_neighbor = data[k];
                hide_features(current_set, feature, possible_nearest_neighbor, directionFlag);
                double distance = euclidean_distance(object_to_classify, possible_nearest_neighbor);

                if (distance < nearest_neighbor_distance){
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k;
                    nearest_neighbor_label = data[k][0];
                }
            }
        }

        //increment number_correctly_classified when the nearest neighbor is in the same class as the current object
        if (label_object_to_classify == nearest_neighbor_label){
            number_correctly_classified++;
        }
    }

    double accuracy = (number_correctly_classified/(static_cast<double> (numRows))) * 100;
    return accuracy;
}

//starts with no features, adds one at a time
void forward_selection (vector <vector<double> > &data, int numColumns, int numRows){
    vector <int> current_features;
    double accuracy = 0;
    double best_accuracy = 0;
    vector <int> best_features;

    cout << "Running nearest neighbor with 0 features, using \"leave-one-out\" evaluation, I get an accuracy of " << default_rate(data, numRows) << "%" << endl;
    cout << "Beginning search.\n" << endl;

    for (int i = 1; i < numColumns; ++i){
        int feature_to_add_at_this_level = 0;
        double best_accuracy_so_far = 0;

        for (int j = 1; j < numColumns; ++j){
            if (find(current_features.begin(), current_features.end(), j) == current_features.end()){
                accuracy = leave_one_out_cross_validation (data, numColumns, numRows, current_features, j, 0);
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

        //keeps track of best feature set overall
        if (best_accuracy == best_accuracy_so_far){
            best_features = current_features;
        }
        
        cout << "\nFeature set {";
            for (int f = 0; f < current_features.size() - 1; ++f){
                cout << current_features.at(f) << ", ";
            } 
        cout << current_features.back() << "} was best, accuracy is " << setprecision(3) << best_accuracy_so_far << "%" << endl << endl;

    }

    cout << "Finished search! The best feature subset is {";
            for (int f = 0; f < best_features.size() - 1; ++f){
                cout << best_features.at(f) << ", ";
            } 
    cout << best_features.back() << "}, which has an accuracy of " << setprecision(3) << best_accuracy << "%" << endl;
}

//starts with all the features, eliminates one at a time
void backward_elimination (vector <vector<double> > &data, int numColumns, int numRows){
    vector <int> current_features;
    double accuracy = 0;
    double best_accuracy = 0;
    vector <int> best_features;
    double defaultRate = default_rate(data, numRows);

    cout << "Beginning search.\n" << endl;

    // load all features into current_features
    for (int i = 1; i < numColumns; ++i){
        current_features.push_back(i);
    }
    
    //tterate through each level of the search tree
    for (int i = 1; i < numColumns; ++i){
        int feature_to_remove_at_this_level = 0;
        double best_accuracy_so_far = 0;

        for (int j = 1; j < numColumns; ++j){
            if (find(current_features.begin(), current_features.end(), j) != current_features.end()){
                //considering removing feature j
                accuracy = leave_one_out_cross_validation (data, numColumns, numRows, current_features, j, 1);
                
                cout << "\tUsing feature(s) {";
                int count = 0;

                for (int f = 0; f < current_features.size(); ++f) {
                    if (current_features[f] != j) {
                        if (count > 0) cout << ", ";
                        cout << current_features[f];
                        count++;
                    }
                } 

                cout << "} accuracy is " << setprecision(3) << accuracy << "%" << endl;
                
                if (accuracy > best_accuracy_so_far){
                    best_accuracy_so_far = accuracy;
                    feature_to_remove_at_this_level = j;
                }
            }
        }

        best_accuracy = max(best_accuracy, best_accuracy_so_far);
        
        //remove feature "feature_to_remove_at_this_level" from current_set
        if (!current_features.empty()){
            vector<int>::iterator it = find(current_features.begin(), current_features.end(), feature_to_remove_at_this_level);
            if (it != current_features.end()) {
                current_features.erase(it);
            }
        }

        //keep track of the best overall accuracy
        if (best_accuracy == best_accuracy_so_far){
            best_features = current_features;
        }

        cout << endl;
        if (!current_features.empty()){
            cout << "Feature set {";
                for (int f = 0; f < current_features.size() - 1; ++f){
                    cout << current_features.at(f) << ", ";
                } 
            cout << current_features.back() << "} was best, accuracy is " << setprecision(3) << best_accuracy_so_far << "%" << endl << endl;
        } else {
            cout << "Feature set {} was best, accuracy is " << defaultRate << "%" << endl << endl;
        }

    }

    //cout << "Running nearest neighbor with all " << numColumns - 1 << " features, using \"leave-one-out\" evaluation, I get an accuracy of " << best_accuracy << "%" << endl;
    cout << "Finished search! The best feature subset is {";
    for (int f = 0; f < best_features.size() - 1; ++f){
        cout << best_features.at(f) << ", ";
    } 
    cout << best_features.back() << "}, which has an accuracy of " << setprecision(3) << best_accuracy << "%" << endl;
}