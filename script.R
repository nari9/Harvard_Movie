#################################################################
# MovieLens Recommendation Script
# 
# Generate movie rating predictions by developing models and test
# on a subset of data and check RMSE on a validation set of data
#################################################################



##################################################
# Create edx set, validation set - supplied by EdX
##################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, movielens, temp, removed)



#######################################################
# install additional libraries not included in EdX code
#######################################################

if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(gridExtra)
library(kableExtra)



##################
# Data Exploration
##################

# overview of training dataset
edx %>% as_tibble()

# number of rows in training set, edx
nrow(edx)

# number of rows in test set, validation
nrow(validation)

# check for any missing values in training set
any(is.na(edx))

#check for any missing values in test set
any(is.na(validation))

# plot of number of movies with count of ratings
movie_count <- edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# plot of number of users with count of ratings
user_count <- edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# place plots side by side
grid.arrange(movie_count, user_count, ncol=2)

# details of top 10 rated movies with more than 100 ratings
# remove kable() to see within R studio
edx %>%
  group_by(movieId, title) %>%
  summarise(Rating_Count = n(),
            Rating_Average = mean(rating)) %>%
  filter(Rating_Count >= 100) %>%
  arrange(desc(Rating_Average)) %>%
  head(10) %>%
  kable()


# top 10 best avg rated movies, no min number of ratings
edx %>%
  group_by(movieId, title) %>%
  summarise(Rating_Count = n(),
            Rating_Average = mean(rating)) %>%
  arrange(desc(Rating_Average)) %>%
  head(10) %>%
  kable() %>%
  kable_styling(full_width = FALSE) %>%
  column_spec(2, width = "20em")


# top 10 worst avg rated movies, no min number of ratings
edx %>%
  group_by(movieId, title) %>%
  summarise(Rating_Count = n(),
            Rating_Average = mean(rating)) %>%
  arrange(Rating_Average) %>%
  head(10) %>%
  kable() %>%
  kable_styling(full_width = FALSE)



################################
# Model 1 - Simple/Average Model
################################

# calculate average rating across all movies, to be used in all other models
mu <- mean(edx$rating)
mu

# define function for RMSE we can use for all models
RMSE <- function(predicted_ratings, actual_ratings){
  sqrt(mean((predicted_ratings - actual_ratings)^2))
}

# calculate RMSE for Model 1
rmse_simple <- RMSE(validation$rating, mu)

# create tibble for calculated RMSE
rmse_model1 <- tibble(Model = "Simple/Average", 
                      Predicted_RMSE = rmse_simple)

# store RMSE for Model 1 to a results table
rmse_results <- rmse_model1

# print table with Model 1 RMSE result
rmse_results %>% kable() %>% 
  kable_styling(full_width = FALSE) %>%
  row_spec(1, bold = TRUE)




######################
# Model 2 - Movie Bias
######################


# calculate movie bias
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# predict ratings including movie bias on validation dataset
predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# calculate the RMSE for model 2
rmse_movie <- RMSE(validation$rating, predicted_ratings)

# create tibble for calculated RMSE
rmse_model2 <- tibble(Model = "Movie Bias Model", 
                      Predicted_RMSE = rmse_movie)

# store RMSE for Model 2 to results table
rmse_results <- bind_rows(rmse_results, 
                          rmse_model2)

# print table with Model 2 RMSE result and previous model results
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  row_spec(2, bold = TRUE)



#############################
# Model 3 - Movie & User Bias
#############################

# calculate user bias using previously calculated mean and movie bias
b_u <- edx %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


# predict ratings including movie and user bias on validation dataset
predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


# calculate the RMSE for model 3
rmse_user <- RMSE(validation$rating, predicted_ratings)


# create tibble for calculated RMSE
rmse_model3 <- tibble(Model = "Movie + User Bias Model", 
                      Predicted_RMSE = rmse_user)


# store RMSE for Model 3 to results table
rmse_results <- bind_rows(rmse_results, rmse_model3)


# print table with Model 3 RMSE result and previous model results
rmse_results %>% knitr::kable() %>%
  kable_styling(full_width = FALSE) %>%
  row_spec(3, bold = TRUE)



##########################################
# Model 4 - Regularization with Movie Bias
##########################################

# Generate a sequence of lambdas
lambdas <- seq(0, 10, 0.1)

# Use different values of lambda on validation dataset to generate predictions
rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the movie bias
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Generate the predicted ratings
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  # Calculate the RMSE
  return(RMSE(validation$rating, predicted_ratings))
})


# plot the result of lambdas against RMSE
qplot(lambdas, rmses)


# lambda value that minimizes the RMSE
min_lambda <- lambdas[which.min(rmses)]
min_lambda


# minimum RMSE value
rmse_reg_movie <- min(rmses)


# create tibble for calculated RMSE
rmse_model4 <- tibble(Model = "Regularized Movie Bias Model", 
                      Predicted_RMSE = rmse_reg_movie)


# store RMSE for Model 4 to results table
rmse_results <- bind_rows(rmse_results, rmse_model4)


# print table with Model 4 RMSE result and previous model results
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  row_spec(4, bold = TRUE)



###################################################
# Model 5 - Regularization with Movie and User Bias
###################################################


# Generate a sequence of lambdas
lambdas <- seq(0, 10, 0.1)

# Use different values of lambda on validation dataset to generate predictions
rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the average of all movies
  mu <- mean(edx$rating)
  
  # Calculate the movie bias
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Calculate the user bias
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
  
  # Generate the predicted ratings
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  # Calculate the RMSE
  return(RMSE(validation$rating, predicted_ratings))
})


# plot the result of lambdas against RMSE
qplot(lambdas, rmses)


# lambda value that minimizes the RMSE
min_lambda <- lambdas[which.min(rmses)]
min_lambda


# minimum RMSE value
rmse_reg_movie_user <- min(rmses)


# create tibble for calculated RMSE
rmse_model5 <- tibble(Model = "Regularized Movie + User Bias Model", 
                      Predicted_RMSE = rmse_reg_movie_user)


# store RMSE for Model 5 to results table
rmse_results <- bind_rows(rmse_results, rmse_model5)


# print table with Model 5 RMSE result and previous model results
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  row_spec(5, bold = TRUE)



#########
# Results
#########


# final results table but highlight RMSE column rather than rows
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  column_spec(2, bold = TRUE)
