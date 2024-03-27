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


# make predictions on test set based on the multinomial logistic regression model 
predictions <- predict(fit, newx = as.matrix(test_images_flat), s = "lambda.min", type = "class")

# Convert test_labels to a factor for consistency in comparison
test_labels_factor <- factor(test_labels + 1) # Adding 1 because R is 1-indexed and class labels start from 1

# Combine predictions and actual labels into a data frame
results_df <- data.frame(Predicted = predictions, Actual = as.character(test_labels_factor))

class_accuracies <- results_df %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(X1 == Actual, na.rm = TRUE))

print(class_accuracies)
###############################GTSRB############################################
library(imager)
library(tidyverse)
library(magrittr)

# Path to the training directory
train_dir <- "D:/GTSRB/Train"

# Function to read, resize, and convert an image to grayscale
process_image <- function(image_path) {
  load.image(image_path) %>%
    resize(32, 32) %>%  # Resize to 32x32 or any desired size
    grayscale() %>%
    as.cimg() %>%
    as.vector()  # Convert the image to a vector
}

# Initialize an empty list to store image data
image_data <- list()

# Initialize an empty vector to store labels
labels <- vector()

# List all subdirectories within the train directory
subdirs <- list.dirs(train_dir, full.names = TRUE, recursive = FALSE)

# Loop through each subdirectory to process images
for (dir in subdirs) {
  # Extract the label from the directory name
  label <- basename(dir)
  
  # List all image files in the subdirectory
  image_files <- list.files(dir, pattern = "\\.png$", full.names = TRUE)
  
  # Process each image and store the data and labels
  for (file in image_files) {
    image_vector <- process_image(file)
    image_data <- c(image_data, list(image_vector))
    labels <- c(labels, label)
  }
}

# Convert the list of image data to a matrix
image_matrix <- do.call(cbind, image_data) %>% t()

# Convert labels to a factor if needed for modeling
labels_factor <- factor(labels)
train_labels_num <- as.numeric(as.character(labels_factor))

# Assuming the environment has sufficient resources to handle this operation
fit <- cv.glmnet(x = as.matrix(image_matrix), y = train_labels_num, family = "multinomial", 
                 parallel = TRUE, trace.it=1)

# Deal with the test set
library(readr)

# Path to the Test.csv file 
test_info <- read_csv("D:/GTSRB/Test.csv")

# Assuming process_image function is defined as before

# Modify this base path according to your specific setup
base_path <- "D:/GTSRB"

# Initialize an empty list to store image data
test_image_data <- list()

# Initialize an empty vector to store labels
test_labels <- vector()

# Process each image
for (i in 1:nrow(test_info)) {
  image_path <- file.path(base_path, test_info$Path[i])
  image_vector <- process_image(image_path)
  test_image_data[[i]] <- image_vector
  test_labels[i] <- as.character(test_info$ClassId[i])
}


# Convert the list of image data to a matrix
test_image_matrix <- do.call(cbind, test_image_data) %>% t()

# Convert labels to numeric
test_labels_factor <- factor(test_labels)
test_labels_num <- as.numeric(as.character(test_labels_factor))


# make predictions on test set based on the multinomial logistic regression model 
predictions <- predict(fit, newx = as.matrix(test_image_matrix), s = "lambda.min", type = "class")

# Convert test_labels to a factor for consistency in comparison
# test_labels_factor <- factor(test_labels + 1) # Adding 1 because R is 1-indexed and class labels start from 1

# Combine predictions and actual labels into a data frame
results_df <- data.frame(Predicted = predictions, Actual = as.character(test_labels_num))

class_accuracies <- results_df %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(X1 == Actual, na.rm = TRUE))

print(class_accuracies, n = 43)

# write out the .csv file
write.csv(class_accuracies, "D:/backdoor_GRASP/GTSRB_test_class_accuracies.csv", row.names = FALSE)

# Define the path to the training data
# setwd <- "D:/GTSRB"
train_path <- "D:/GTSRB/Train"

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

# 
