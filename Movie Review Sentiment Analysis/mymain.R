# Package Preration----------------------------------------------------------
packages = c('text2vec', 'glmnet', 'data.table', 'tm', 'wordcloud')
for (pkg in packages){
  if (!require(pkg, character.only = TRUE)){
    install.packages(pkg)
  }
}

library(text2vec)
library(data.table)
library(glmnet)
library(tm)
setwd("/Users/SHEN/Desktop/Movie_Review_Sentiment_Analysis")

# Data Clean --------------------------------------------------------------
train <- read.delim("labeledTrainData.tsv", quote = "", as.is = T)
test <- read.delim("testData.tsv", quote = "", as.is = T)

allpred <- function (x){
  x$review <- gsub("<.*?>", "", x$review)
  x$review <- tolower(gsub("[[:punct:]0-9[:blank:]]+", " ", x$review))
  setDT(x)
  prep_fun <- tolower
  tok_fun <- word_tokenizer
  #Transfer to matrix
  it <- itoken(x$review, 
              preprocessor = prep_fun, 
              tokenizer = tok_fun, 
              ids = x$id, 
              progressbar = FALSE)
  return(it)
}

it_train <- allpred(train)
it_test <- allpred(test)

#Create bag of words
vocab <- create_vocabulary(it_train, ngram = c(1L, 2L))

vocab <- vocab %>% prune_vocabulary(term_count_min = 7, 
                                   doc_proportion_max = 0.6,
                                   doc_proportion_min = 0.001)

bigram_vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, bigram_vectorizer)


# Lasso model -------------------------------------------------------------
NFOLDS <- 6
glmnet_classifier <- cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = NFOLDS,
                              thresh = 1e-4,
                              maxit = 1e3)

print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
dtm_test <- create_dtm(it_test, bigram_vectorizer)
preds <- predict(glmnet_classifier,s = exp(-6), dtm_test, type = 'class')
prob <- cbind(test$id, preds)
colnames(prob) <- c("id", "sentiment")
write.csv(prob, "mySubmition.csv",row.names = F,quote = FALSE, sep = ",")


# Data Visualization ------------------------------------------------------
newtest <- cbind(test, preds)
colnames(newtest) <- c("id","review","sentiment")

newtest <- newtest[1:50,]

#Create bag of words without stopwords
stop_words = stopwords()
vocab2 <- create_vocabulary(it_train,stopwords =  stop_words)

vocab2 <- vocab2 %>% prune_vocabulary(term_count_min = 7, 
                                    doc_proportion_max = 0.6,
                                    doc_proportion_min = 0.001)

bigram_vectorizer2 <- vocab_vectorizer(vocab2)
dtm_train2 <- create_dtm(it_train, bigram_vectorizer2)

#fit new model with new bag of words
NFOLDS <- 6
glmnet_classifier2 <- cv.glmnet(x = dtm_train2, y = train[['sentiment']], 
                               family = 'binomial', 
                               alpha = 1,
                               type.measure = "auc",
                               nfolds = NFOLDS,
                               thresh = 1e-4,
                               maxit = 1e3)

dtm_test2 <- create_dtm(it_test, bigram_vectorizer2)
predtest <- predict(glmnet_classifier2,s = exp(-6), dtm_test2, type = 'coefficients')[,1]
predtest <- predtest[-1]

positive =predtest[which(predtest > 0)]
negative =predtest[which(predtest < 0)]
poswords = names(positive)
negwords = names(negative)
# remove HTML tags    
newtest[,2] = gsub("<.*?>", " ", newtest[,2])
newtest[,2] = gsub("\\\\", " ", newtest[,2])
myfile = "sentiment_output.html"
if (file.exists(myfile)) file.remove(myfile)
n.review = dim(newtest)[1]

## create html file
write(paste("<html> \n", 
            "<head> \n",  
            "<style> \n",
            "@import \"textstyle.css\"", 
            "</style>", 
            "</head> \n <body>\n"), file=myfile, append=TRUE)
write("<ul>", file=myfile, append=TRUE)

for(i in 1:n.review){
  write(paste("<li><strong>", newtest[i,1], 
              "</strong> sentiment =", newtest[i,3], 
              "<br><br>", sep=" "),
        file=myfile, append=TRUE)
  tmp = strsplit(newtest[i,2], " ")[[1]]
  tmp.copy = tmp
  nwords = length(tmp)
  negid = which(!is.na(match(negwords, strsplit(newtest[i,2], " ")[[1]])))
  posid = which(!is.na(match(poswords, strsplit(newtest[i,2], " ")[[1]])))
  wordlist.neg=negwords[negid]
  wordlist.pos=poswords[posid]
  
  neg=NULL;
  if(length(wordlist.neg)>0){
  for(j in 1:length(wordlist.neg))
    neg = c(neg, grep(wordlist.neg[j], tmp, ignore.case = TRUE))
  }
  if (length(neg)>0) {
    for(j in 1:length(neg)){
      tmp.copy[neg[j]] = paste("<span class=\"neg\">", 
                               tmp.copy[neg[j]], "</span>", sep="")
    }
  }
  
  pos=NULL;
  if(length(wordlist.pos)>0){
  for(j in 1:length(wordlist.pos))
    pos = c(pos, grep(wordlist.pos[j], tmp, ignore.case = TRUE))
  }
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"pos\">", 
                               tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  write( paste(tmp.copy, collapse = " "), file=myfile, append=TRUE)
  write("<br><br>", file=myfile, append=TRUE)
}

write("</ul> \n  </body> \n </html>", file=myfile, append=TRUE)


# Word Cloud
pocoef = unname(positive)
potable = data.frame(poswords,pocoef)
newpotable = potable[order(potable$pocoef,decreasing = TRUE),][1:100,]
necoef = unname(negative)
netable = data.frame(negwords,necoef)
newnetable = netable[order(netable$necoef,decreasing = FALSE),][1:100,]
newnetable$necoef = abs(newnetable$necoef)
newnetable = newnetable[-which(newnetable$negwords == "mst"),]

library(wordcloud)
wordcloud(newpotable$poswords, newpotable$pocoef,random.order=FALSE, rot.per=.25, 
          max.words = 50,scale=c(3,.15),colors=brewer.pal(8,"Dark2"))
wordcloud(newnetable$negwords, newnetable$necoef,random.order=FALSE, rot.per=.25, 
          max.words = 50,scale=c(2,.1),colors=brewer.pal(8,"Dark2"))

