# Load libraries
install.packages("RWeka")
install.packages("ggsci")

system("sudo apt-get -y install libmagick++-dev", intern=TRUE)
install.packages("magick", verbose=TRUE)

# Load libraries
library(readr)
library(tidyverse)
library(tm)
library(wordcloud)
library(wordcloud2)
library(tidytext)
library(textdata)
library(reshape2)
library(RWeka)
library(knitr)
library(gridExtra)
library(grid)
library(magick)
library(igraph)
library(ggraph)
library("ggsci")
library(devtools)
library(circlize)
library(radarchart)


# Read the data 
scripts <- read.csv("D:/SKRIPSI/data baru/low.csv", row.names = NULL, sep = ";", encoding = "latin1")

# Read the Lexicons (for sentiment classification)
bing <- read.csv("C:/Users/HP/Downloads/Bing.csv")
afinn <- read.csv("C:/Users/HP/Downloads/Afinn.csv")
nrc <- read.csv("C:/Users/HP/Downloads/NRC.csv")

cleanCorpus <- function(text){
  # punctuation, whitespace, lowercase, numbers
  text.tmp <- tm_map(text, removePunctuation)
  text.tmp <- tm_map(text.tmp, stripWhitespace)
  text.tmp <- tm_map(text.tmp, content_transformer(tolower))
  text.tmp <- tm_map(text.tmp, removeNumbers)
  
  # removes stopwords
  stopwords_remove <- c(stopwords("en"), c("thats","weve","hes","theres","ive","im",
                                           "will","can","cant","dont","youve","us",
                                           "youre","youll","theyre","whats","didnt"))
  text.tmp <- tm_map(text.tmp, removeWords, stopwords_remove)
  
  return(text.tmp)
}


frequentTerms <- function(text){
  
  # create the matrix
  s.cor <- VCorpus(VectorSource(text))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  s.tdm <- removeSparseTerms(s.tdm, 0.999)
  m <- as.matrix(s.tdm)
  word_freqs <- sort(rowSums(m), decreasing = T)
  
  # change to dataframe
  dm <- data.frame(word=names(word_freqs), freq=word_freqs)
  
  return(dm)
}


# Bigram tokenizer
tokenizer_2 <- function(x){
  NGramTokenizer(x, Weka_control(min=2, max=2))
}

# Bigram function 
frequentBigrams <- function(text){
  
  s.cor <- VCorpus(VectorSource(text))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl, control=list(tokenize=tokenizer_2))
  s.tdm <- removeSparseTerms(s.tdm, 0.999)
  m <- as.matrix(s.tdm)
  word_freqs <- sort(rowSums(m), decreasing=T)
  dm <- data.frame(word=names(word_freqs), freq=word_freqs)
  
  return(dm)
}


# Trigram tokenizer
tokenizer_3 <- function(x){
  NGramTokenizer(x, Weka_control(min=3, max=3))
}

# Trigram function 
frequentTrigrams <- function(text){
  
  s.cor <- VCorpus(VectorSource(text))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl, control=list(tokenize=tokenizer_3))
  s.tdm <- removeSparseTerms(s.tdm, 0.999)
  m <- as.matrix(s.tdm)
  word_freqs <- sort(rowSums(m), decreasing=T)
  dm <- data.frame(word=names(word_freqs), freq=word_freqs)
  
  return(dm)
}


# Top 15 characters with the most dialogues
scripts %>% 
  # prepare the table
  count(character) %>%
  arrange(desc(n)) %>% 
  slice(1:15) %>%
  
  # the plot
  ggplot(aes(x=reorder(character, n), y=n)) +
  geom_bar(stat="identity", aes(fill=n), show.legend=F) +
  geom_label(aes(label=n)) +
  scale_fill_gradient(low="#58D68D", high="#239B56") +
  labs(x="Character", y="Number of dialogues", title="Top 15 Characters with the Most Dialogues") +
  coord_flip() +
  theme_bw()

# Creating our tokens
tokens <- scripts %>% 
  mutate(dialogue = as.character(scripts$dialogue)) %>% 
  unnest_tokens(word, dialogue)

tokens %>% head(5) %>% select(character, word)


tokens %>% 
  # append the bing sentiment and prepare the data
  inner_join(bing, "word") %>%
  count(word, sentiment, sort=T) %>% 
  acast(word ~ sentiment, value.var = "n", fill=0) %>% 
  
  # wordcloud
  comparison.cloud(colors=c("#991D1D", "#327CDE"), max.words = 100)

to_plot <- tokens %>% 
  # get 'bing' and filter the data
  inner_join(bing, "word") %>% 
  filter(character %in% c("Cooper","Cobb","Joker","Brand","Dent")) %>% 
  
  # sum number of words per sentiment and character
  count(sentiment, character) %>% 
  group_by(character, sentiment) %>% 
  summarise(sentiment_sum = sum(n)) %>% 
  ungroup()

### VISUALISASI NRC
sentiments <- tokens %>% 
  inner_join(nrc, "word") %>%
  count(sentiment, sort=T)

sentiments

# The plot:
sentiments %>% 
  ggplot(aes(x=reorder(sentiment, n), y=n)) +
  geom_bar(stat="identity", aes(fill=sentiment), show.legend=F) +
  geom_label(label=sentiments$n) +
  labs(x="Sentiment", y="Frequency", title="What is the general atmosphere of Lowest Rated English Movie?") +
  coord_flip() + 
  theme_bw() +
  
  # rick and morty customized scale ^^
  scale_fill_rickandmorty(palette = c("schwifty"), alpha = 0.8)


### VISUALISASI AFINN
tokens %>% 
  # Count how many word per value
  inner_join(afinn, "word") %>% 
  count(value, sort=T) %>%
  
  # Plot
  ggplot(aes(x=value, y=n)) +
  geom_bar(stat="identity", aes(fill=n), show.legend = F, width = 0.5) +
  geom_label(aes(label=n)) +
  scale_fill_gradient(low="#85C1E9", high="#3498DB") +
  scale_x_continuous(breaks=seq(-5, 5, 1)) +
  labs(x="Score", y="Frequency", title="Word count distribution over intensity of sentiment: Neg -> Pos") +
  theme_bw()

tokens %>% 
  # Count how many word per value
  inner_join(afinn, "word") %>% 
  count(value, sort = TRUE) %>%
  
  # Plot
  ggplot(aes(x = value, y = n)) +
  geom_bar(stat = "identity", aes(fill = n), show.legend = FALSE, width = 0.5) +
  geom_label(aes(label = n)) +
  scale_fill_gradient(low = "#85C1E9", high = "#3498DB") +
  scale_x_continuous(breaks = seq(-5, 5, 1)) +
  labs(x = "Score", y = "Frequency", title = "Word count distribution over intensity of sentiment: Neg -> Pos") +
  theme_bw()

wordcloud2(frequentTerms(scripts$dialogue), size=1.6, minSize = 0.9, color='random-light', 
           backgroundColor="black", shape="diamond", fontFamily="HersheySymbol")

