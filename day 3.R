#liner regression: starbucks - amount vs. cups + days 
fit2 = lm(Starbucks$Amount ~ Starbucks$Cups + Starbucks$Days)
res = residuals(fit2)

summary(fit2)