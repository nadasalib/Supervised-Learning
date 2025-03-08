#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

double leave_one_out_cross_validation(vector <vector<double> > &data, vector <int>, int);
void feature_search(vector <vector<double> > &data, int numColumns);

int main(){

    string fileName;
    cout << "Type in the name of the file to test: ";
    cin >> fileName;

    cout << "Type in the number of the algorithm you want to run." << endl << endl;
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
    cout << "Running nearest neighbor with all " << numColumns - 1<< " features, using \"leave-one-out\" evaluation, I get an accuracy of [ACCURACY]%" << endl << endl;

    cout << "Beginning search." << endl;

    feature_search(data, numColumns);

}

//function stub returns random accuracy to test feature_search
double leave_one_out_cross_validation(vector <vector<double> > &data, vector <int> current_set, int feature_to_add){
    int min = 0.00;
    int max = 100.00;
    int randomNumber = min + (rand() % (max - min + 1));
    return randomNumber;
}

void feature_search (vector <vector<double> > &data, int numColumns){
    vector <int> current_features;
    int accuracy = 0;

    for (int i = 1; i < numColumns; ++i){
        cout << "On the " << i << "th level of the search tree" << endl;
        int feature_to_add_at_this_level = 0;
        int best_accuracy_so_far = 0;

        for (int j = 1; j < numColumns; ++j){
            if (find(current_features.begin(), current_features.end(), j) == current_features.end()){
                cout << "--Considering adding the " << j << " feature" << endl;
                    accuracy = leave_one_out_cross_validation (data, current_features, j + 1);
                
                if (accuracy > best_accuracy_so_far){
                    best_accuracy_so_far = accuracy;
                    feature_to_add_at_this_level = j;
                }
            }
        }

        current_features.push_back(feature_to_add_at_this_level);
        cout << "On level " << i << " I added feature " << feature_to_add_at_this_level << " to current set" << endl;
    }
}

//CS170_Small_Data__87.txt
//CS170_Large_Data__123.txt