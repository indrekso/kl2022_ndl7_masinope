# Sotsiaalse analüüsi meetodid: kvantitatiivne lähenemine
# Otsustuspuu: hüperparameetrid ja pesastatud ristvalideerimine paketiga mlr
# aluseks on võetud näide raamatust Rhys, H. (2020) Machine Learning with R, the tidyverse, and mlr. Manning Publications. 
# 6. mai 2022

library(mlr)
library(mlbench)
library(tidyverse)

data(Zoo, package = "mlbench")

zooTib <- as_tibble(Zoo)
zooTib

zooTib <- mutate_if(zooTib, is.logical, as.factor)
zooTib

# paneme paika ülesande ehk defineerime andmed ja klassifitseerimistunnuse ning algoritmi

zooTask <- makeClassifTask(data = zooTib, target = "type")
tree <- makeLearner("classif.rpart")

# uurime, millised hüperparameetrid on otsustuspuu puhul võimalikud

getParamSet(tree)

# kui tahame hüperparameetreid tuunida, on vaja ette anda eelnevad objektid (andmed ja algoritm) ning võimalike hüperparameetrite loend, kuidas nende hulgast hüperparameetrite väärtusi valime ning valideerimise eeskiri.
 
treeParamSpace <- makeParamSet(
  makeIntegerParam("minsplit", lower = 5, upper = 20),
  makeIntegerParam("minbucket", lower = 3, upper = 10),
  makeNumericParam("cp", lower = 0.01, upper = 0.1),
  makeIntegerParam("maxdepth", lower = 3, upper = 10))

randSearch <- makeTuneControlRandom(maxit = 200)
cvForTuning <- makeResampleDesc("CV", iters = 5)

# optimeerime hüperparameetreid

tunedTreePars <- tuneParams(tree, task = zooTask,
                            resampling = cvForTuning,
                            par.set = treeParamSpace,
                            control = randSearch)

tunedTreePars

# optimeeritud mudeli treenimine eelnevalt leitud parimate hüperparameetrite põhjal

tunedTree <- setHyperPars(tree, par.vals = tunedTreePars$x)
tunedTreeModel <- train(tunedTree, zooTask)

# otsustuspuu visualiseerimine

library(rpart.plot)

treeModelData <- getLearnerModel(tunedTreeModel)

rpart.plot(treeModelData, 
           box.palette = "BuBn",
           type = 5)

# vaatame mudelit lähemalt

printcp(treeModelData, digits = 3)

# pesastatud valideerimine - optimaalsete hüperparameetrite otsing ja selle valideerimine kaasatakse mudeli valideerimisse, valideeritakse kogu mudeli koostamise protsessi integreeritud tervikuna. Võimaldab saada realistlikuma hinnangu mudeli täpsusele uute andmete korral

# paneme paika üldise valideerimisskeemi (5-fold CV)

outer <- makeResampleDesc("CV", iters = 5)

# defineerime õpimeetodi, kuhu on kaasatud mudeli optimeerimine ehk hüperparameetrite otsing. Nagu näha, on ka siin seadistatud valideerimine (antud juhul samamoodi 5-fold CV), mida kasutatakse üldise valideerimisskeemi treeningandmetel.

treeWrapper <- makeTuneWrapper("classif.rpart", resampling = cvForTuning,
                               par.set = treeParamSpace,
                               control = randSearch)

# viime läbi mudeli treenimise koos optimeerimisega. Üldise valideerimisskeemi järgi jaotatakse andmestik viieks osaks, st andmestik jaotatakse viis korda treeningvalimiks (80% andmestikust) ja testvalimiks (20%), iga treeningvalimit kasutatakse hüperparameetrite tuunimiseks, kus omakorda viiakse läbi valideerimine (st 80% algandmestikust jagatakse omakorda viieks osaks). Saadud hüperparameetrite kombinatsiooni kasutatakse mudeli treenimiseks üldise treeningvalimi peal, mida testitakse üldise testvalimi peal. Viie mudeli pealt saadud MMCE-dest võetakse keskmine.

cvWithTuning <- resample(treeWrapper, zooTask, resampling = outer)

cvWithTuning
