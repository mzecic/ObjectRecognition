#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


void computeConfusionMatrix(const std::string& eval_file) {
    // Define the classes in a fixed order
    std::vector<std::string> classes = {"wallet", "mouse", "shoe", "pen", "watch"};

    // Create 5x5 confusion matrix initialized to zeros
    std::vector<std::vector<int>> confusion_matrix(5, std::vector<int>(5, 0));

    // Read evaluation data
    std::ifstream infile(eval_file);
    std::string line;

    // Skip header if exists
    while (std::getline(infile, line)) {
        std::istringstream ss(line);
        std::string true_label, pred_label;

        // Read true and predicted labels
        std::getline(ss, true_label, ',');
        std::getline(ss, pred_label, ',');

        // Remove any whitespace
        true_label.erase(remove_if(true_label.begin(), true_label.end(), isspace), true_label.end());
        pred_label.erase(remove_if(pred_label.begin(), pred_label.end(), isspace), pred_label.end());

        // Find indices for labels
        auto true_idx = std::find(classes.begin(), classes.end(), true_label) - classes.begin();
        auto pred_idx = std::find(classes.begin(), classes.end(), pred_label) - classes.begin();

        // Increment the appropriate cell
        if (true_idx < classes.size() && pred_idx < classes.size()) {
            confusion_matrix[true_idx][pred_idx]++;
        }
    }

    // Calculate total samples and correct predictions
    int total_samples = 0;
    int correct_predictions = 0;
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            total_samples += confusion_matrix[i][j];
            if (i == j) {
                correct_predictions += confusion_matrix[i][j];
            }
        }
    }

    // Print the confusion matrix
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "Predicted →\n";

    // Print header
    std::cout << "True ↓    ";
    for (const auto& cls : classes) {
        std::cout << std::setw(8) << cls;
    }
    std::cout << "\n";

    // Print matrix with row labels
    for (size_t i = 0; i < 5; i++) {
        std::cout << std::setw(9) << classes[i];
        for (size_t j = 0; j < 5; j++) {
            std::cout << std::setw(8) << confusion_matrix[i][j];
        }
        std::cout << "\n";
    }

    // Print accuracy
    double accuracy = static_cast<double>(correct_predictions) / total_samples * 100;
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(2)
              << accuracy << "%" << std::endl;

    // Print per-class statistics
    std::cout << "\nPer-class Statistics:\n";
    for (size_t i = 0; i < 5; i++) {
        int class_total = 0;
        for (size_t j = 0; j < 5; j++) {
            class_total += confusion_matrix[i][j];
        }
        if (class_total > 0) {
            double class_accuracy = static_cast<double>(confusion_matrix[i][i]) / class_total * 100;
            std::cout << classes[i] << " Accuracy: " << std::fixed << std::setprecision(2)
                      << class_accuracy << "%" << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <evaluation_file.csv>" << std::endl;
        return 1;
    }

    std::string eval_file = argv[1];

    computeConfusionMatrix(eval_file);
}
