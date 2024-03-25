library(keras)
library(glmnet)
library(dplyr)

###############################CIFAR-10########################################
# Load the CIFAR-10 dataset
cifar <- dataset_cifar10()
c(train_images, train_labels) %<-% cifar$train
c(test_images, test_labels) %<-% cifar$test

# Inspect the data
dim(train_images)  # Should show (50000, 32, 32, 3)
dim(test_images)   # Should show (10000, 32, 32, 3)

# Extract the first 5000 and 1000 observations from train and test data set
#train_images <- train_images[1:5000, , , ]
#train_labels <- train_labels[1:5000, ]
#test_images <- test_images[1:1000, , , ]
#test_labels <- test_labels[1:1000, ]

# Flatten the images
train_images_flat <- array_reshape(train_images, c(dim(train_images)[1], 32*32*3))
test_images_flat <- array_reshape(test_images, c(dim(test_images)[1], 32*32*3))

# Convert labels to a binary matrix
train_labels_matrix <- to_categorical(train_labels, 10)
test_labels_matrix <- to_categorical(test_labels, 10)


# Assuming your environment has sufficient resources to handle this operation
fit <- cv.glmnet(x = as.matrix(train_images_flat), y = train_labels+1, family = "multinomial", parallel = TRUE, trace.it=1)

# make predictions on test set based on the multinomial logistic regression model 
predictions <- predict(fit, newx = as.matrix(test_images_flat), s = "lambda.min", type = "class")

# Convert test_labels to a factor for consistency in comparison
test_labels_factor <- factor(test_labels + 1) # Adding 1 because R is 1-indexed and class labels start from 1

# Combine predictions and actual labels into a data frame
results_df <- data.frame(Predicted = predictions, Actual = test_labels_factor)

# Calculate accuracy for each class
class_accuracies <- results_df %>%
  +   group_by(Actual) %>%
  +   summarise(Accuracy = mean(X1 == Actual))

print(class_accuracies)
###############################MNIST############################################ 
mnist <- dataset_mnist()
c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

# Flatten the images
train_images_flat <- array_reshape(train_images, c(dim(train_images)[1], 28*28))
test_images_flat <- array_reshape(test_images, c(dim(test_images)[1], 28*28))

# Scale the data to [0, 1]
train_images_flat <- train_images_flat / 255
test_images_flat <- test_images_flat / 255

# Convert labels to a binary matrix
train_labels_matrix <- to_categorical(train_labels, 10)
test_labels_matrix <- to_categorical(test_labels, 10)

# Use a subset of the data for a quick test
#sub_train_images_flat <- train_images_flat[1:50000, ]
#sub_train_labels <- train_labels[1:50000]

# Convert labels to numeric if they are factors
#sub_train_labels_numeric <- as.numeric(as.character(sub_train_labels))+1
train_labels_numeric <- as.numeric(as.character(train_labels))+1

# Fit the model on the subset
#fit <- cv.glmnet(x = as.matrix(sub_train_images_flat), y = sub_train_labels_numeric, family = "multinomial",
#                 parallel = TRUE, trace.it=1)


# Assuming your environment has sufficient resources to handle this operation
fit <- cv.glmnet(x = as.matrix(train_images_flat), y = train_labels_numeric, family = "multinomial", 
                 parallel = TRUE, trace.it=1)


# Predictions
predictions <- predict(fit, newx = as.matrix(test_images_flat), s = "lambda.min", type = "class")
test_labels_factor <- factor(test_labels + 1) # Adjust if necessary
accuracy <- mean(predictions == test_labels_factor)
print(paste("Accuracy:", accuracy))
###############################GTSRB############################################
library(imager)
library(tidyverse)

# Define the path to the training data
setwd <- "D:/GTSRB"
train_path <- "Train"

# Function to read images from a subfolder and assign labels
read_images_with_labels <- function(subfolder_name) {
  full_path <- file.path(train_path, subfolder_name)
  image_files <- list.files(path = full_path, full.names = TRUE)
  image_data <- lapply(image_files, load.image)
  image_labels <- rep(subfolder_name, length(image_data))
  
  data_frame(images = image_data, labels = image_labels)
}

# Get a list of subfolders (each representing a label)
labels <- list.dirs(path = train_path, full.names = FALSE, recursive = FALSE)

# Read images and labels from each subfolder
dataset <- bind_rows(lapply(labels, read_images_with_labels), .id = "label")