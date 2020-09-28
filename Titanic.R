# Loading requried packages

library(dplyr)        # Data Manipulation
library(ggplot2)      # Data Visualization
library(Amelia)       # Missing value plot
library(mice)         # Multivariate Imputation
library(randomForest) # Random Forests for Classification and Regression
library(stringr)      # String Operation
library(ROCR)         # ROC Plot
library(pROC)         # AUC calculation

# Imorting the datasets

train <- read.csv("E:/vallisnaria/internship_iprimed_7thsem/projectcodeanddatasets/train.csv", header = TRUE)
test <- read.csv("E:/vallisnaria/internship_iprimed_7thsem/projectcodeanddatasets/test.csv", header = TRUE)
train <- dplyr::select(train, c(PassengerId, Pclass:Embarked, Survived))

full <- dplyr::bind_rows(train, test)

# Let's take a peek at dataset

dim(full)
str(full)      # dplyr::glimpse(train)
summary(full)

# Proportion of male and female

round(prop.table(table(train$Sex))*100)

# Feature engineering: Calculate the family size

full$Fam_size <- full$SibSp + full$Parch + 1


# Find the family based on last names
str_split(full$Name[1],",")
full$Surname <- sapply(full$Name, function(x) str_split(x, pattern = ",")[[1]][1])
summary(full$Fam)

# Create a family variable
full$Family <- paste(full$Surname, full$Fam_size, sep='_')

# Survival penalty

ggplot(data = full[1:891, ], aes(x = Fam_size, fill = factor(Survived))) +
  geom_bar(stat = "count", position = "dodge") +
  scale_x_continuous(breaks = c(1:11)) +
  labs(x = "Family size", y = "Count")

# Create a Deck variable. Get passenger deck A - F:

full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1])) # Using strsplit (instead of str_split). Otherwise output is NA.

# Exploring little more

hist(full$Fare)
hist(full$Age)

barplot(table(train$Pclass))

table(train$Sex, train$Pclass)

prop.table(table(train$Sex, train$Pclass))
prop.table(table(train$Sex, train$Pclass), 1)
prop.table(table(train$Sex, train$Pclass), 2)

train %>%
  group_by(Sex) %>%
  summarize(sum = sum(Fare), n = n())

train %>%
  group_by(Sex) %>%
  filter(Survived == "1") %>%
  summarize(sum = n())

table(train$Survived)

length(train$Survived[train$Survived == "1"])

sum(train$Survived=="1")

# Missing value treatment

summary(full)  # There are two passangers whose port of embarkation is unknown. See row number 62, and 830

mis_embark <- full[c(62, 830), ]

# Let's visualize embarkment, passanger class, and median fare

ggplot(data = full[-c(62, 830), ], aes(x = Embarked, y = Fare, fill = Pclass)) +
  geom_boxplot() +
  geom_hline(aes(yintercept = 80), colour = "red", linetype = 1, lwd = 1.5)

# You see that $80 is median fare for passangers with Pclass = 1 and port of embarkment as C
# That means we can consider "C" to be value for embark, for the row no 62, 830

full[c(62, 830), ]$Embarked <- "C"

# Time to drop the level ("") from the factor (Embarked)

full$Embarked <- droplevels(full$Embarked)#not actually required

# Check again for missing values

summary(full)
which(!complete.cases(full$Fare)) # to get the row number where fare is missing

full[1044, ]

# Get rid of NA value for Fare for row 1044

ggplot(full[full$Pclass == "3" & full$Embarked == "S", ], aes(x = Fare)) +
  geom_density(fill = "cyan2", alpha = 0.4) +
  geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), colour = "red", linetype = 2, lwd = 0.5) +
  scale_x_continuous(limits = c(0, 85))

#legends are in introduced if you use colur in aesthetics, you can also use a variable to introduce a third variable 
#ggplot(full[full$Pclass == "3" & full$Embarked == "S", ], aes(x = Fare,fill = "cyan2", alpha = 0.4)) +
#geom_density() +
#geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), colour = "red", linetype = 2, lwd = 0.5) +
#scale_x_continuous(limits = c(0, 85))


full[1044, ]$Fare <- median(full[full$Pclass == "3" & full$Embarked == "S", ]$Fare, na.rm = TRUE)

# Imputing Ages

# Number of missing Age values
sum(is.na(full$Age))

# Make variables factors into factors
factor_vars <- c("PassengerId", "Pclass", "Sex", "Embarked", "Surname", "Family")

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf')

# Save the complete output
mice_output <- complete(mice_mod)

#	Let us compare the results we get with the original distribution of passenger ages to ensure that nothing has gone completely awry.

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data',
     col='slategray1', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output',
     col='steelblue4', ylim=c(0,0.04))

#	Things look good, so let us replace our age vector in the original data with the output from the mice model.

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

# Prediction

# Split the data back into a train set and a test set
train_final <- full[1:891,]
test_final <- full[892:1309,]

# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)

log_model <- glm(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Fam_size, data = train_final, family = binomial)
log_model
summary(log_model)


# Prediction on training set

train_final$predicted <- ifelse(predict(log_model, type = "response") > 0.5, 1, 0)

# To see how the model predicted on training data

caret::confusionMatrix(data = factor(train_final$predicted), reference = factor(train_final$Survived))

# Plot ROC curve

pred <- prediction(train_final$predicted, train_final$Survived)
perf <- performance(pred,"tpr","fpr")
plot(perf)

# Calculate AUC

auc(roc(train_final$Survived, train_final$predicted))

# Prediction on test set

test_final$predicted <- ifelse(predict(log_model, newdata = test_final, type = "response") > 0.5, 1, 0)

table(test_final$predicted)
