##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# End of given code
##########################################################

# Work with the edx dataset to build our model

# mu: average rating of all movies
mu<- mean(edx$rating)

# Average movie ratings histogram
edx %>% 
  group_by(movieId) %>%
  summarize(movie_average = mean(rating)) %>%
  ggplot(aes(movie_average)) +
  geom_histogram(bins = 35, color= "black") 

# b_i: parameter that accounts for movie bias
b_i <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Average user ratings histogram
edx %>% 
  group_by(userId) %>% 
  summarize(user_average = mean(rating)) %>% 
  ggplot(aes(user_average)) + 
  geom_histogram(bins = 35, color = "black")

# b_u: parameter that accounts for user bias
b_u <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Genre average ratings histogram
edx %>% 
  group_by(genres) %>%
  summarize(genre_average = mean(rating)) %>%
  ggplot(aes(genre_average)) +
  geom_histogram(bins= 25, color = "black")

# b_g: parameter that accounts for genre bias
b_g <- edx %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

### y_hat: predicted ratings for the validation set

y_hat <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by="genres")%>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

## Ratings range from 0.5 to 5 in the edx dataset

# Find which predicted values are lower than 0.5 and replace them with 0.5
ind_lower<- which(y_hat<0.5)
y_hat[ind_lower]<- 0.5 

# Find which predicted values are higher than 5 and replace them with 5
ind_over<- which(y_hat>5)
y_hat[ind_over]<- 5

# Create a function to compute the RMSE
RMSE<- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Compute RMSE
RMSE(validation$rating, y_hat)
