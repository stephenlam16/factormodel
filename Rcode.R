
library(sparseDFM)

data <- read.csv('M:/Team 3/team projects/2024/Pension funds diversifying portoflio/dynamic factor/analysis/code/stockdatanew_Rtrial.csv')



my_array <- array(
  data = c(unlist(data), unlist(data)),
  dim = c(221, 45),
  dimnames = list(rownames(data), colnames(data))
)

tuneFactors(my_array, type = 2, standardize = TRUE,
            r.max = min(15,ncol(data)-1), plot = TRUE)

fit.sdfm <- sparseDFM(my_array, r = 2, alphas = logspace(-2,3,100),
                      alg = 'EM-sparse', err = 'IID', kalman = 'univariate')


summary(fit.sdfm)

fit.sdfm$em$alpha_opt

plot(fit.sdfm, type = 'factor')

plot(fit.sdfm, type = 'residual', use.series.names = TRUE)

plot(fit.sdfm, type = 'loading.heatmap', use.series.names = TRUE)

write.csv(residuals(fit.sdfm),'D:/backup/intdiv/trial/residual.csv')
write.csv(fitted.values(fit.sdfm),'D:/backup/intdiv/trial/fitted.csv')
