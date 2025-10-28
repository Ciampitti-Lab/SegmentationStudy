# Gustavo's data

# Setup -------------------------------------------------------------------

library(readxl)
library(dplyr)
library(tidyr)
library(tibble)
library(magrittr)
library(tidyr)
library(ggplot2)
library(ggpmisc)
library(gridExtra)
library(extrafont)
library(ggtext)
library(egg)
library(car)
library(emmeans)
# library(marginaleffects)
library(betareg)
library(lmtest)

theme_set(theme_bw())
zmargin <- theme(panel.spacing = unit(0,"lines"))



# Data import -------------------------------------------------------------

data <- read.csv("c:/Users/Federico/Downloads/fede_play_data.csv")

str(data)

data$collectiondate %<>% as.factor()
data$genotype %<>% as.factor()

summary(data)
data$iout <- ( data$IoU * (length(data$IoU) - 1) + 0.5 ) / length(data$IoU)
data$recallt <- ( data$Recall * (length(data$Recall) - 1) + 0.5 ) / length(data$Recall)


table(data$genotype, data$collectiondate)
# another way to remove 0s and 1s from the variable.
eps <- 0.0001
data$iout <- pmin(pmax(data$IoU, eps), 1 - eps)

# Analysis ----------------------------------------------------------------

m.io_1 <- betareg(IoU ~ collectiondate * genotype, data = data)
m.io_2 <- betareg(IoU ~ collectiondate + genotype, data = data)

m.io_0 <- betareg(iout ~ 1, data = data)
m.io_1 <- betareg(iout ~ collectiondate * genotype, data = data)
m.io_2 <- betareg(iout ~ collectiondate + genotype, data = data)
m.io_3 <- betareg(iout ~ genotype, data = data)
m.io_4 <- betareg(iout ~ collectiondate, data = data)

lrtest(m.io_2, m.io_3) # p < 2.2e-16 (collection date effect)
lrtest(m.io_2, m.io_4) # p < 2.2e-16 (genotype effect)

m.io_2_em.gen <- emmeans(m.io_2, ~ genotype)
m.io_2_em.ps <- emmeans(m.io_2, ~ collectiondate)

summary(pairs(m.io_2_em.gen), type = "response", infer = c(T, T))
m.io_2_em.gen %>% multcomp::cld(Letters = letters, sort = F, type = "response", adjust = "none")


