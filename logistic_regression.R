library(keras)

# Load the CIFAR-10 dataset
cifar <- dataset_cifar10()
c(train_images, train_labels) %<-% cifar$train
c(test_images, test_labels) %<-% cifar$test

# Inspect the data
dim(train_images)  # Should show (50000, 32, 32, 3)
dim(test_images)   # Should show (10000, 32, 32, 3)

# Flatten the images
train_images_flat <- array_reshape(train_images, c(dim(train_images)[1], 32*32*3))
test_images_flat <- array_reshape(test_images, c(dim(test_images)[1], 32*32*3))

# Convert labels to a binary matrix
train_labels_matrix <- to_categorical(train_labels, 10)
test_labels_matrix <- to_categorical(test_labels, 10)

library(glmnet)

# Assuming your environment has sufficient resources to handle this operation
fit <- cv.glmnet(x = as.matrix(train_images_flat), y = train_labels+1, family = "multinomial")
# make predictions on test set based on the multinomial logistic regression model 
predictions <- predict(fit, newx = as.matrix(test_images_flat), s = "lambda.min", type = "class")


library(dplyr)

# Convert test_labels to a factor for consistency in comparison
test_labels_factor <- factor(test_labels + 1) # Adding 1 because R is 1-indexed and class labels start from 1

# Combine predictions and actual labels into a data frame
results_df <- data.frame(Predicted = predictions, Actual = test_labels_factor)

# Calculate accuracy for each class
class_accuracies <- results_df %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))

print(class_accuracies)