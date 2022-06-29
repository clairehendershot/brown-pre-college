#liner regression: starbucks - amount vs. cups + days 
fit2 = lm(Starbucks$Amount ~ Starbucks$Cups + Starbucks$Days)
res = residuals(fit2)

summary(fit2)

# 1. liner regression: NBA - all
fit3 = lm(NBA$Wins ~ NBA$`Field Goal%` + NBA$`Field Goal % Allowed` + NBA$`Field Goal % Difference`)
res = residuals(fit3)

summary(fit3)

# 2. for every win -> 4.84% increase of field goals
# 3. y = 41.64 + 4.84(field goal %) + -4.86(field goal % allowed)
#    y = 41.64 + 217.8 - 213.84
#      = 45.6
# 4. yes adam ploted a qq plot and the graph displayed a positive linear relationship
# 5. yes there is a significant relationship. p-value (1.799e-08) < 0.05
# 6. p-value = 1.799e-08. probability distribution will be less than the observed result (0.05)
# 7. -
# 8. adjusted r-squared = 0.7134

# liner regression: NBA - field goal % difference
fit4 = lm(NBA$Wins ~ NBA$`Field Goal % Difference`)
res = residuals(fit4)

summary(fit4)

# liner regression: NBA - field goal %
fit5 = lm(NBA$Wins ~ NBA$`Field Goal%`)
res = residuals(fit5)

summary(fit5)

# liner regression: NBA - field goal % allowed
fit6 = lm(NBA$Wins ~ NBA$`Field Goal % Allowed`)
res = residuals(fit6)

summary(fit6)

# 9. the most appropriate regression model is the one with the wins vs. field goal % difference
# 10. field goal % difference: 1.603e-09
#     field goal %: 2.382e-05
#     field goal % allowed: 9.469e-06