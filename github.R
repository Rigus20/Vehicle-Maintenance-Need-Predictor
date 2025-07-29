# Load base R libraries for logistic regression inference and cross-validation
library(car)              # bring in ellipse functions
library(leaps)            # subset selection
library(BAS)              # Bayesian Adaptive Sampling
library(faraway)          # supporting regression tools
library(rgl)              # 3D plotting
library(corrplot)         # correlation heatmaps

# Load libraries for graphics and data wrangling
library(GGally)           # ggpairs(), imports ggplot2
library(dplyr)            # data manipulation
library(readxl)           # Excel import

# Load libraries for generalized linear models and diagnostics
library(rms)              # lrm(), residuals methods
library(arm)              # model diagnostics
library(ResourceSelection) # Hosmer–Lemeshow goodness-of-fit test
library(pROC)             # ROC curves
library(PRROC)           # precision–recall curves
library(boot)             # bootstrap
library(ROSE)             # oversampling techniques
library(caret)            # model training and tuning

# Import dataset from Excel and display dimensions and summary
data <- read_excel("data.xlsx")

dim(data)
summary(data)

# Convert categorical variables to factors with specified levels
data$Fuel_Type            <- factor(data$Fuel_Type, levels = c("Petrol", "Diesel", "Electric"))
data$Maintenance_History  <- factor(data$Maintenance_History, levels = c("Poor", "Average", "Good"))
data$Transmission_Type    <- factor(data$Transmission_Type, levels = c("Automatic", "Manual"))
data$Owner_Type           <- factor(data$Owner_Type, levels = c("First", "Second", "Third"))
data$Tire_Condition       <- factor(data$Tire_Condition, levels = c("Worn Out", "Good", "New"))
data$Brake_Condition      <- factor(data$Brake_Condition, levels = c("Worn Out", "Good", "New"))
data$Battery_Status       <- factor(data$Battery_Status, levels = c("Weak", "Good", "New"))

# Convert factor variables to numeric for analysis
data$Fuel_Type           <- as.numeric(data$Fuel_Type)
data$Maintenance_History <- as.numeric(data$Maintenance_History)
data$Transmission_Type   <- as.numeric(data$Transmission_Type)
data$Owner_Type          <- as.numeric(data$Owner_Type)
data$Tire_Condition      <- as.numeric(data$Tire_Condition)
data$Brake_Condition     <- as.numeric(data$Brake_Condition)
data$Battery_Status      <- as.numeric(data$Battery_Status)

# Verify variable class and type, and inspect data
class(data$Fuel_Type)
typeof(data$Fuel_Type)
View(data)

# Balance the response variable by undersampling to avoid overfitting
data_0 <- data[data$Need_Maintenance == 0, ]
data_0_under <- data_0[sample(1:500, 500), ]
data_1 <- data[data$Need_Maintenance == 1, ]
data_1_under <- data_1[sample(1:500, 500), ]
data <- rbind(data_1_under, data_0_under)

# Build logistic regression model and evaluate with BIC
g <- glm(Need_Maintenance ~ ., data = data, family = binomial(link = logit))
summary(g)
BIC(g)

# Remove non-significant variable and compare BIC
g <- update(g, . ~ . - Maintenance_History)
summary(g)
BIC(g)

# Reinstate full model and perform stepwise selection based on AIC
g <- glm(Need_Maintenance ~ ., data = data, family = binomial(link = logit))
modello_stepwise <- step(g, direction = "both")
summary(modello_stepwise)



# Fit logistic regression model using selected covariates
g <- glm(data$Need_Maintenance ~ Mileage + Maintenance_History + 
           Reported_Issues + Vehicle_Age + Fuel_Type + Owner_Type + 
           Service_History + Accident_History + Fuel_Efficiency + Brake_Condition + 
           Battery_Status, data = data, family = binomial(link = logit))

# Open new graphics window and display pairwise variable relationships
dev.new()
ggpairs(data[, c('Mileage', 'Maintenance_History', 
                 'Reported_Issues', 'Vehicle_Age', 'Fuel_Type', 'Owner_Type', 
                 'Service_History', 'Accident_History', 'Fuel_Efficiency', 'Brake_Condition', 
                 'Battery_Status')], aes(col = as.factor(data$Need_Maintenance)))

## Examine each covariate's relationship with response variable
# Plot Need_Maintenance vs. Mileage
# Calculate group means of Need_Maintenance by mileage bins
min(data$Mileage)
max(data$Mileage)
summary(data$Mileage)
x <- c(30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000)
mid <- (x[-1] + x[-length(x)]) / 2
GRP <- cut(data$Mileage, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
# Plot raw data and overlay group means
plot(data$Mileage, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Mileage', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Mileage', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Maintenance_History
min(data$Maintenance_History)
max(data$Maintenance_History)
summary(data$Maintenance_History)
x <- c(1, 2, 3)
mid <- c(1.5, 2.5)
GRP <- cut(data$Maintenance_History, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Maintenance_History, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Maintenance_History', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Maintenance_History', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Reported_Issues
min(data$Reported_Issues)
max(data$Reported_Issues)
summary(data$Reported_Issues)
x <- 0:5
mid <- (x[-1] + x[-length(x)]) / 2
GRP <- cut(data$Reported_Issues, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Reported_Issues, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Reported_Issues', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Reported_Issues', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Vehicle_Age
min(data$Vehicle_Age)
max(data$Vehicle_Age)
summary(data$Vehicle_Age)
x <- 1:10
mid <- (x[-1] + x[-length(x)]) / 2
GRP <- cut(data$Vehicle_Age, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Vehicle_Age, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Vehicle_Age', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Vehicle_Age', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Fuel_Type
min(data$Fuel_Type)
max(data$Fuel_Type)
summary(data$Fuel_Type)
x <- 1:3
mid <- c(1.5, 2.5)
GRP <- cut(data$Fuel_Type, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Fuel_Type, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Fuel_Type', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Fuel_Type', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Owner_Type
min(data$Owner_Type)
max(data$Owner_Type)
summary(data$Owner_Type)
x <- 1:3
mid <- c(1.5, 2.5)
GRP <- cut(data$Owner_Type, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Owner_Type, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Owner_Type', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Owner_Type', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Service_History
min(data$Service_History)
max(data$Service_History)
summary(data$Service_History)
x <- 1:10
mid <- (x[-1] + x[-length(x)]) / 2
GRP <- cut(data$Service_History, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Service_History, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Service_History', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Service_History', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Accident_History
min(data$Accident_History)
max(data$Accident_History)
summary(data$Accident_History)
x <- 0:3
mid <- (x[-1] + x[-length(x)]) / 2
GRP <- cut(data$Accident_History, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Accident_History, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Accident_History', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Accident_History', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Fuel_Efficiency
min(data$Fuel_Efficiency)
max(data$Fuel_Efficiency)
summary(data$Fuel_Efficiency)
x <- 0:20
mid <- (x[-1] + x[-length(x)]) / 2
GRP <- cut(data$Fuel_Efficiency, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Fuel_Efficiency, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Fuel_Efficiency', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Fuel_Efficiency', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Brake_Condition
min(data$Brake_Condition)
max(data$Brake_Condition)
summary(data$Brake_Condition)
x <- 1:3
mid <- c(1.5, 2.5)
GRP <- cut(data$Brake_Condition, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Brake_Condition, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Brake_Condition', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Brake_Condition', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

# Plot Need_Maintenance vs. Battery_Status
min(data$Battery_Status)
max(data$Battery_Status)
summary(data$Battery_Status)
x <- 1:3
mid <- c(1.5, 2.5)
GRP <- cut(data$Battery_Status, breaks = x, include.lowest = TRUE, right = FALSE)
y <- tapply(data$Need_Maintenance, GRP, mean)
plot(data$Battery_Status, data$Need_Maintenance,
     pch = ifelse(data$Need_Maintenance == 1, 3, 4),
     col = ifelse(data$Need_Maintenance == 1, 'forestgreen', 'red'),
     xlab = 'Battery_Status', ylab = 'Need_Maintenance',
     main = 'Need_Maintenance vs. Battery_Status', lwd = 2, cex = 1.5)
points(mid, y, pch = 16)

## Check multicollinearity using variance inflation factors
vif(g)


# Compute odds ratios for a 10-unit change in each predictor
exp(10 * coef(g)[2])  # estimate OR for Mileage per 10-unit increase
exp(10 * coef(g)[3])  # estimate OR for Maintenance_History per 10-unit increase
exp(10 * coef(g)[4])  # estimate OR for Reported_Issues per 10-unit increase
exp(10 * coef(g)[5])  # estimate OR for Vehicle_Age per 10-unit increase
exp(10 * coef(g)[6])  # estimate OR for Fuel_Type per 10-unit increase
exp(10 * coef(g)[7])  # estimate OR for Owner_Type per 10-unit increase
exp(10 * coef(g)[8])  # estimate OR for Service_History per 10-unit increase
exp(10 * coef(g)[9])  # estimate OR for Accident_History per 10-unit increase
exp(10 * coef(g)[10]) # estimate OR for Fuel_Efficiency per 10-unit increase
exp(10 * coef(g)[11]) # estimate OR for Brake_Condition per 10-unit increase
exp(10 * coef(g)[12]) # estimate OR for Battery_Status per 10-unit increase

# Generate predicted probabilities and plot binned residuals
predicted_values <- predict(g, type = "link")  # obtain linear predictors
predicted_probs <- plogis(predicted_values)      # transform to probability scale
binnedplot(predicted_probs, rstandard(g),      
           main = "Binned residuals plot")      

# Perform Hosmer–Lemeshow goodness-of-fit test
hoslem.test(g$y, fitted(g), g = 11)

# Calculate and plot ROC curve for model performance
ROC <- PRROC::roc.curve(scores.class0 = g$fitted.values, 
                        weights.class0 = data$Need_Maintenance, 
                        curve = TRUE)
plot(ROC, color = TRUE)

# Leave One Out Cross-Validation

# Split the dataset into training (80%) and test (20%) sets
train_index <- createDataPartition(data$Need_Maintenance, p = 0.8, list = FALSE, times = 1)
train_set <- data[train_index, ]
test_set <- data[-train_index, ]

# Fit logistic regression model on training set
g_train <- glm(Need_Maintenance ~ Mileage + Maintenance_History + Reported_Issues + Vehicle_Age + Fuel_Type + Owner_Type + Service_History + Accident_History + Fuel_Efficiency + Brake_Condition + 
                 Battery_Status, data = train_set, family = binomial(link = logit))
# Compute LOOCV error for the fitted model
cv.err <- cv.glm(train_set, g_train)
cv.err$delta[1]

# Fit baseline model using all predictors and compute its LOOCV error
g0_train <- glm(Need_Maintenance ~ ., data = train_set, family = binomial(link = logit))
cv.err_0 <- cv.glm(train_set, g0_train)
cv.err_0$delta[1]

# Explore higher polynomial degrees up to 4 for selected predictors
n <- 4
N_M <- train_set
for (p in 2:n) {
  # Generate p-th degree terms for each predictor
  N_M <- cbind(N_M, N_M$Mileage^p)
  N_M <- cbind(N_M, N_M$Maintenance_History^p)
  N_M <- cbind(N_M, N_M$Reported_Issues^p)
  N_M <- cbind(N_M, N_M$Vehicle_Age^p)
  N_M <- cbind(N_M, N_M$Fuel_Type^p)
  N_M <- cbind(N_M, N_M$Owner_Type^p)
  N_M <- cbind(N_M, N_M$Service_History^p)
  N_M <- cbind(N_M, N_M$Accident_History^p)
  N_M <- cbind(N_M, N_M$Fuel_Efficiency^p)
  N_M <- cbind(N_M, N_M$Brake_Condition^p)
  N_M <- cbind(N_M, N_M$Battery_Status^p)
}

# Initialize vector to store LOOCV errors for each degree
cv.error <- rep(0, n)
for (p in 1:n) {
  # Fit model including terms up to degree p and perform stepwise selection
  mod_p <- glm(Need_Maintenance ~ ., data = N_M[, 2:(20 + 11 * (p - 1))], family = binomial(link = logit))
  mod_p <- mod_p %>% stepAIC(trace = FALSE)
  # Compute LOOCV error for the selected model
  cv.error[p] <- cv.glm(N_M[, 1:(20 + 11 * (p - 1))], mod_p)$delta[1]
}

# Report pre-computed LOOCV errors for degrees 1 to 4
#cv.error <- c(0.10053722, 0.02061726, 0.02056157, 0.03224512)
# Plot LOOCV error versus polynomial degree
plot(1:n, cv.error, type = 'b', xlab = "Degree of Polynomial", ylab = "LOOCV error")

# Determine classification threshold based on training set performance
threshold <- 0.5
real_values <- train_set$Need_Maintenance
predicted_probs <- 1 / (1 + exp(- predict.glm(g_train, train_set)))
predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)
confusion_matrix <- table(predicted_classes, real_values)
confusion_matrix

# Update threshold to 0.4 for improved performance
threshold <- 0.4

# Make predictions on the test set using the chosen threshold
predicted_values <- predict.glm(g_train, test_set)
predicted_probs <- 1 / (1 + exp(-predicted_values))
predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)

# Compute performance metrics on test set
real_values <- test_set$Need_Maintenance
confusion_matrix <- table(predicted_classes, real_values)
TP <- confusion_matrix[2, 2]
FP <- confusion_matrix[2, 1]
FN <- confusion_matrix[1, 2]
TN <- confusion_matrix[1, 1]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
accuracy <- (TP + TN) / (TP + TN + FP + FN)

# Display evaluation metrics
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1_score, "\n")
cat("Accuracy:", accuracy, "\n")
