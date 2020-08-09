library(tidyverse)
library(tidytext)

# https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-05-05

# load the data
user_reviews <- read_tsv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/user_reviews.tsv")

# do the data preprocessing
reviews <- user_reviews %>%
  mutate(rating = if_else(grade > 7, "good", "bad"),
         rating_bad = as.numeric(rating == "bad"),
         text = str_remove(text, "Expand"),
         text_length = str_length(text),
         phrase = str_to_lower(text) %>% 
           str_detect("(1|one) island (per|for) (console|switch|nintendo)"))

# histogram of grade
reviews %>% ggplot(aes(x=grade)) + geom_histogram()

# plot timeline of pos/neg reviews
reviews %>% 
  count(date, rating) %>%
  ggplot(aes(x = date, y = n, color = rating)) +
  geom_line()

# plot text length and ocurrence of "the phrase"
reviews %>% 
  ggplot(aes(x = log(text_length), y = grade, color = phrase)) +
  geom_point() +
  geom_smooth()

# extract all words
review_words <- reviews %>%
  unnest_tokens(word, text, "words")

# get the sentiment lexicon
sentiments <- get_sentiments("afinn")

# join sentiments
review_words <- review_words %>% 
  left_join(sentiments)

# compute mean sentiment score by review
reviews <- review_words %>% 
  group_by(user_name) %>%
  summarize(mean_sentiment = mean(value, na.rm = TRUE)) %>% 
  replace_na(list(mean_sentiment = 0)) %>% 
  right_join(reviews)

# create n-grams
grams <- reviews %>%
  unnest_tokens(grams, text, "ngrams", n = 3) %>%
  mutate(grams = fct_lump_n(factor(grams), n = 20))

# turn n-grams into "features" (columns) 
grams <- grams %>% 
  filter(grams != "Other") %>%
  drop_na(grams) %>%
  mutate(grams = paste0("ngram_", grams)) %>%
  count(user_name, grams) %>%
  pivot_wider(id_cols = user_name,
              names_from = grams,
              values_from = n)
grams[is.na(grams)] <- 0 # trick to replace all NA with 0

# join the n-gram features with the original data
reviews <- reviews %>% 
  left_join(grams)
reviews[is.na(reviews)] <- 0 # trick to replace all NA with 0

# create the standardaize function that we need later for 
# mutating all regression inputs (in the mutate_at call)
standardize <- function(x){
  z <- (x - mean(x)) / sd(x)
  return(z)
}

# set up a tibble with all regression inputs 
reviews_reg <- reviews %>%
  select(rating_bad, mean_sentiment, phrase, text_length, starts_with("ngram_")) %>%
  mutate(text_length = log(text_length),
         phrase = as.numeric(phrase)) %>%
  mutate_at(vars(-rating_bad), .funs = standardize)

# calculate basline accuracy
mean(reviews_reg$rating_bad)

# run the logistic regression
logit_regression <- glm(rating_bad ~ ., data = reviews_reg)

# load the broom package to tidy up regression output
library(broom)

# tidy coefficients with confidence intervals
logit_coef <- tidy(logit_regression, conf.int = TRUE)

# plot text features (only)
logit_coef %>% 
  filter(term != "(Intercept)", !str_detect(term, "ngram_")) %>%
  ggplot(aes(x = term, y = estimate)) +
  geom_point() +
  geom_linerange(aes(ymin=conf.low, ymax=conf.high)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_flip()

# plot n-grams (only)
logit_coef %>% 
  filter(str_detect(term, "ngram_")) %>%
  mutate(term = fct_reorder(factor(term), estimate)) %>%
  ggplot(aes(x = term, y = estimate)) +
  geom_point() +
  geom_linerange(aes(ymin=conf.low, ymax=conf.high)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_flip()

# plot n-grams and text features (color n-grams)
logit_coef %>% 
  filter(term != "(Intercept)") %>%
  mutate(term = fct_reorder(factor(term), estimate),
         ngram = str_detect(term, "ngram")) %>%
  ggplot(aes(x = term, y = estimate, color = ngram)) +
  geom_point() +
  geom_linerange(aes(ymin=conf.low, ymax=conf.high)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_flip()

### EXTRA (NOT COVERED IN VIDEO)
# I realized that I also wanted to 
# show you how to acces the accuracy 
# of the model.
# Note, that you need to specify 
#   type = "response" 
# in the predict functioon to get the 
# predicted probabilities!

reviews_augmented <- reviews %>% 
  mutate(prediction_prob = predict(logit_regression, type = "response"),
         prediction = round(prediction_prob),
         prediction_correct = (rating_bad == prediction))

# In-sample accuracy is roughly 75%. Not that bad!
mean(reviews_augmented$prediction_correct)

# We can look how well we predict each outcome...
reviews_augmented %>%
  group_by(rating_bad) %>%
  summarise(accuracy = mean(prediction_correct))

# Seems like we detect negative reviews quite well,
# but do not predict positive reviews that well.

# You can investigate what's going on and maybe
# come ups with more features to llok at.
# Have look at all reviews that were predicted
# incorrectly:
reviews_augmented %>%
  filter(!prediction_correct)

# You can also work on making the graphs look
# prettier, or make the logit model a bit more
# sophisticated...
# Or go on to a new project! Just keep on learning! ;)

