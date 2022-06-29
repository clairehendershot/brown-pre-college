install.packages("ggplot2")
library(ggplot2)
titanic = read.csv('titanic.csv', stringsAsFactors=FALSE)

titanic$Pclass = factor(titanic$Pclass)
titanic$Survived = factor(titanic$Survived)


summary(titanic)
str(titanic)

#black and white graph
ggplot(titanic, aes(x=Pclass)) + geom_bar()

# loading in built-in datasets 
mydata <- mtcars[, c(1,3,4,5,6,7)]
head(mydata)
cormat <- round(cor(mydata), 2)
head(cormat)

library(readxl)
dominos <- read_excel("Documents/MTL 2700/Resurants Drive Thru.xlsx")
View(dominos)

# question 2
# t-test, p-value <= 0.05 --> null hypothesis
t.test(dominos$Time, alternative = "less", mu=173.62)

#question 3
# t-test, p-value <= 0.05 --> null hypothesis
t.test(dominos$Time, alternative = "two.sided", mu=173.62)

