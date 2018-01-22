# install.package("DataExplorer")
# install.package("data.table")
# install.package("randomForest")
# install.package("caret")
# install.package("glmnet")

setwd("/Users/SHEN/Desktop/House_Prices")
Iowa <- read.csv('train.csv')
set.seed(1)

alpha <- 0.75
idx <- sample(1:nrow(Iowa), alpha * nrow(Iowa))
train0 <- Iowa[idx, ]
test0 <- Iowa[-idx, ] 
dim(train0)
dim(test0)

Iowa_pro <- function(train, test){
  
  timestart <- Sys.time()
  
  # Write a function named data_clean for data cleaning
  data_clean <- function(train1, test1){
    
    # Combine train and test first for data cleaning
    Iowa <- rbind(train1, test1)
    
    # Find rows of train and test for future splitting after data cleaning
    n_train <- dim(train1)[1]
    n_test <- dim(test1)[1]
    
    # Find integer variables with less than 15 unique values and convert them to factors 
    a <- sapply((sapply(Iowa[, sapply(Iowa, is.integer)], unique)), length)
    b <- a[a <= 15]
    cols <- names(b)
    Iowa[cols] <- lapply(Iowa[cols], factor)
    
    # Convert varibles "OverallCond" and "OverallQual" into ordered factor
    Iowa$OverallCond <- ordered(Iowa$OverallCond, levels = paste(1:10, sep = "")) 
    Iowa$OverallQual <- ordered(Iowa$OverallQual, levels = paste(1:10, sep = "")) 
    
    # Checking missing data
    numNA <- colSums(apply(Iowa[, -c(1, 81)], 2, is.na))
    number_of_missing <- numNA[which(numNA != 0)]               # # of NA's
    data_type <- sapply(Iowa[,names(which(numNA != 0))], class) # Data type
    
    # Removing variables with missing values more than 1000
    Iowa <- Iowa[ , !(names(Iowa) %in% names(which(number_of_missing > 1000)))]
    
    # Add new level NA for missing values for factor variables
    temp_cat0 <- names(which(data_type == "factor"))
    temp_cat <- temp_cat0[!temp_cat0 %in% names(which(number_of_missing > 1000))]
    for (j in temp_cat){
      Iowa[ ,j] <- addNA(Iowa[ ,j])  # addNA treat the NA's as a new level called '<NA>'
    }
    
    # Using median to replace missing values for integer variables
    temp_num <- names(which(data_type == "integer"))
    for (j in temp_num){
      na.id <- is.na(Iowa[, j])                     # binary indicator: NA (1) or not (0)
      tempMedian <- median(Iowa[, j], na.rm = TRUE) # find the median
      Iowa[which(na.id), j] <- tempMedian
    }
    
    # Drop variables with single level larger than 1300
    data.type <- sapply(Iowa[, -c(1, ncol(Iowa))], class)
    t1 <- sapply(Iowa[names(Iowa)[which(c(NA, data.type, NA) == 'factor')]], table)
    Iowa <- Iowa[ , !(names(Iowa) %in% names(t1[sapply(t1, max) >= 1300]))]
    
    data.type <- sapply(Iowa[, -c(1, ncol(Iowa))], class)
    t2 <- sapply(Iowa[names(Iowa)[which(c(NA, data.type, NA) == 'factor')]], table)
    
    # Set the level of the variable "FireplaceQu"
    levels(Iowa$FireplaceQu) <- c("Ex", "Fa", "Gd", "Po", "TA", "Missing" )
    
    library(DataExplorer)  # Used for reconstructing factor levels
    library(data.table)    # Used for changing data structure
    
    Iowa <- data.table(Iowa)
    name <- names(t2) 
    
    # Reconstructing levels for factors, combine levels with less than 10% frequency in to a new level, "other"
    for (i in 1:length(name)){
      temp <- name[i]
      CollapseCategory(Iowa, temp, 0.1, update = T)
    }
    
    # Convert data structure back to data frame
    Iowa <- data.frame(Iowa) 
    
    # Convert data type back to factors
    for(i in colnames(Iowa[,sapply(Iowa, is.character)])){
      Iowa[,i] <- as.factor(Iowa[,i])
    }
    
    data.type <- sapply(Iowa[, -c(1, ncol(Iowa))], class)
    cat_var <- names(Iowa)[which(c(NA, data.type, NA) == 'factor')]
    num_var <-  names(Iowa)[which(c(NA, data.type, NA) == 'integer')]
    t3 <- sapply(Iowa[cat_var], table)
    
    # Scale all remaining continuous variables
    Iowa[, num_var] <- scale(Iowa[, num_var])
    
    # Applied log transformation on SalePrice
    Iowa$SalePrice <- log(Iowa$SalePrice + 1)
    
    # Done data cleaning
    # Split data back to train and test, based on the index
    train2 <- Iowa[1:n_train,]
    test2 <- Iowa[(n_train + 1) : dim(Iowa)[1],]
    
    return(list(train2, test2))
  }
  
  # Apply data clean function to both train and test data
  clean_Iowa <- data_clean(train, test)
  
  train <- as.data.frame(clean_Iowa[[1]])
  test <- as.data.frame(clean_Iowa[[2]])
  
  # RMSE for evaluting accuracy
  RMSE <- function(x,y){
    a <- sqrt(sum((log(x)-log(y))^2)/length(y))
    return(a)
  }
  
  ####################################################################################
  ### Linear Regression ###
  #########################
  data.type <- sapply(train[, -c(1, ncol(train))], class)
  num_var <-  names(train)[which(c(NA, data.type, NA) == 'numeric')]
  
  corr <- cor(train[, c(num_var, 'SalePrice')])  # correlation matrix
  
  highCor <- which(abs(corr[, ncol(corr)]) > 0.5)
  highCor <- highCor[-length(highCor)]
  linear_pred = names(highCor)
  
  # variables with high corr with saleprice are stored in linear_pred
  pred <- paste(linear_pred, collapse= "+")
  lm_fit <- as.formula(paste("SalePrice ~ ", pred, collapse= "+"))
  fit1 <- lm(lm_fit, data = train)
  
  yhat1 <- exp(predict(fit1, newdata = test))-1 
  RMSE1 <- RMSE(yhat1, exp(test$SalePrice)-1)
  
  submission1 = matrix(c(test$Id,yhat1),length(yhat1),2)
  write.table(submission1, 'mysubmission1.txt', row.names = F, col.names=c("ID","Prediction"))
  
  ####################################################################################
  ### Random Forest ###
  #####################
  library(randomForest)
  fit2 <- randomForest(SalePrice~., data = train,
                       method = 'anova',
                       ntree = 300,
                       mtry = floor(dim(train)[2]/3), #p/3,p is the number of variables
                       replace = F,
                       nodesize = 5,
                       improtance = F)
  
  yhat2 <- exp(predict(fit2, newdata = test))-1 
  RMSE2 <- RMSE(yhat2, exp(test$SalePrice)-1)
  
  # Get the importance of each variable and plot it.
  importance(fit2)
  varImpPlot (fit2, n.var=15)
  
  submission2 = matrix(c(test$Id,yhat2),length(yhat2),2)
  write.table(submission2, 'mysubmission2.txt', row.names = F, col.names=c("ID","Prediction"))
  
  ####################################################################################
  ### Lasso ###
  #############
  library(caret)
  library(glmnet)
  
  # Transform category to dummy variables
  data.type <- sapply(train[, -c(1, ncol(train))], class)
  cat_var <-  names(train)[which(c(NA, data.type, NA) == 'factor')] 
  
  # For train data
  dummies_train <- dummyVars(~.,train[cat_var])
  categorical_train_hot <- predict(dummies_train,train[cat_var])
  categorical_train_hot[is.na(categorical_train_hot)] <- 0 
  traindata <- cbind(train[num_var],categorical_train_hot,train$SalePrice)
  
  # For test data
  dummies_test <- dummyVars(~.,test[cat_var])
  categorical_test_hot <- predict(dummies_test,test[cat_var])
  categorical_test_hot[is.na(categorical_test_hot)] <- 0 
  testdata <- cbind(test[num_var],categorical_test_hot,test$SalePrice)
  
  # Set the design matrix and response variable
  traindata <- apply(traindata,2,as.numeric)
  X <- traindata[,-dim(traindata)[2]]
  Y <- traindata[,dim(traindata)[2]]
  X.test=testdata[,-dim(traindata)[2]]
  
  # Build the lasso model
  fit3 <- cv.glmnet(X, Y, alpha = 1)
  lambda.min <- fit3$lambda.min
  yhat3 <- exp(predict(fit3, newx = as.matrix(X.test),
                       s = lambda.min, type = "class"))-1
  RMSE3 <- RMSE(yhat3, exp(testdata[, dim(testdata)[2]])-1)
  
  submission3 = matrix(c(test$Id,yhat3),length(yhat3),2)
  write.table(submission3, 'mysubmission3.txt', row.names = F, col.names=c("ID","Prediction"))
  
  
  ####################################################################################
  
  res = matrix(c(RMSE1, RMSE2, RMSE3), 3, 1,
         dimnames = list(c("Linear Regression", "RandomForest", "Lasso"),
                         c("RMSE")))
  print(res)
  
  timeend<-Sys.time()
  runningtime<-timeend-timestart
  return(c(RMSE1, RMSE2, RMSE3,runningtime))
  
}

# Iowa_pro(train0, test0)


sd <- matrix(rep(0,20), 5, 4)

for(i in 1:5){
  
  alpha <- 0.75
  idx <- sample(1:nrow(Iowa), alpha * nrow(Iowa))
  train0 <- Iowa[idx, ]
  test0 <- Iowa[-idx, ] 
  
  sd[i,] <- Iowa_pro(train0, test0)
}

apply(sd[,1:3], 2, sd)
apply(sd, 2, mean)



