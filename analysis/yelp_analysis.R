library(dplyr)
X = read.csv('/home/gavagai/Dropbox/X.csv')

X = select(X, -(review_text))
X = select(X, -(X))	

X$week_of_year = as.factor(X$week_of_year)

logit.overall = glm(label ~ . - reviewer_state - reviewer_location,
                    family = "binomial",
                    data = X)



model.empty= glm(label ~ 1,
            family = "binomial",
            data = X)

library(MASS) #The Modern Applied Statistics library.
scope = list(lower = formula(model.empty), upper = formula(logit.overall))

#Stepwise regression using AIC as the criteria (the penalty k = 2).
forwardAIC = step(model.empty, scope, direction = "forward", k = 2)
backwardAIC = step(logit.overall, scope, direction = "backward", k = 2)
bothAIC.empty = step(model.empty, scope, direction = "both", k = 2)
bothAIC.full = step(logit.overall, scope, direction = "both", k = 2)

1 - logit.overall$deviance/logit.overall$null.deviance
