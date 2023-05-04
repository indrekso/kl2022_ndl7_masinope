# Sotsiaalse analüüsi meetodid: kvantitatiivne lähenemine
# Masinõpe KNN näitel paketiga mlr
# 4. mai 2023

library(dplyr)
library(mlr)

# loeme sisse osa Chapel Hill Expert Survey 2014 andmestikus Euroopa parlamentide erakondade kohta (Ryan Bakker, Erica Edwards, Liesbet Hooghe, Seth Jolly, Gary Marks, Jonathan Polk, Jan Rovny, Marco Steenbergen, and Milada Vachudova. 2015. “2014 Chapel Hill Expert Survey.”); h/t: Bruno Castanho Silva @ ECPR Winter School 2020.

df <- read.csv("data/ches_2014.csv")

# vaatame tunnuste jaotusparameetreid

psych::describe(df)

# koostame kNN meetodil mudeli, mis klassifitseerib erakonna vasak- või parempoolseks

df <- df %>%
  mutate(lrgen = ifelse(lrgen < 5, 'left','right')) %>% 
  mutate(lrgen = as.factor(lrgen))

# klassifitseerimine toimub kahe tunnuse alusel:
# 
# spendvtax = position on improving public services vs. reducing taxes.
# 0 = Fully in favour of raising taxes to increase public services
# 10 = Fully in favour of cutting public services to cut taxes
# 
# immigrate_policy = position on immigration policy.
# 0 = Fully opposed to a restrictive policy on immigration
# 10 = Fully in favor of a restrictive policy on immigration

# vaatame, kuidas need tunnused on omavahel ja vasak-parempoolsusega seotud

ggplot(df, aes(spendvtax, immigrate_policy, colour = lrgen)) +
  geom_point()

# lihtsalt huvi pärast vaatame ka Eesti erakondade paigutust

df %>% filter(cname == "est") %>% 
  ggplot(aes(spendvtax, immigrate_policy, colour = lrgen, label = party_name)) +
  geom_point() +
  geom_label() +
  xlim(0, 100) +
  ylim(0, 10)

# spendvatx ja immigrate_policy on erinevatel skaaladel, tuleks standardiseerida. Antud juhul sobiks standardiseerimiseks spendvtax skaala jagamine 10-ga, sellisel juhul on skaalad ka sisuliselt paremini tõlgendatavad.

df <- df %>% 
  mutate(spendvtax = spendvtax / 10)

# saanuks siiski ka z-skoorideks teisendamist kasutada
# df <- df %>% 
#  mutate(across(where(is.numeric), scale))

# koostame mudeli, kus võtame k väärtuseks esmalt 17 (lähim paaritu arv ruutjuurele erakondade arvust andmestikus) ja testime selle täpsust. Selleks eraldame andmed mudeli koostamiseks (n-ö treenimiseks/õpetamiseks) ja testimiseks. Juhuslikustasin juba eelnevalt ridade järjekorra andmestikus, nii et saame siin andmed kaheks eraldada lihtsalt reanumbrite järgi. 

df <- df %>% 
  select(lrgen, spendvtax, immigrate_policy)

train <- df %>% 
  slice(1:179) 

test <- df %>% 
  slice(180:268)

# kasutame masinõppe meetodeid paketi mlr abil, millega saab R-s hõlpsalt KNN-i ja muid masinõppemeetodeid (sh tavalisi regressioonimudeleid) kasutada. Teised sarnased n-ö katuspaketid R-s on caret ja tidymodels. Paketi mlr loogika erinevate masinõppemeetodite puhul on, et kõigepealt paneme paika ülesande, mida tahame masinõppega lahendada - antud juhul teatud kahe tunnuse põhjal klassifitseerida erakonnad vasak- ja parempoolseteks. 

lrTask <- makeClassifTask(data = train, target = "lrgen")

# defineerime õpimeetodi ehk valime algoritmi, mida klassifitseerimisel kasutame koos vajalike argumentidega

knn <- makeLearner("classif.knn", k = 17)

# infoks: milliseid õpimeetodeid mlr veel võimaldab?

listLearners()$class
listLearners("classif")$class

# loome mudeli - rakendame eelnevalt defineeritud meetodi sõnastatud ülesandele ja saame mudeli, millega on pmst võimalik tulevikus uute andmete põhjal järeldusi teha (uute erakondade puhul otsustada, kas tegu on vasak- või parempoolse erakonnaga). Mudeli koostamine käib paketis mlr funktsiooniga train, mille esimene argument on õpimeetod (vastav objekt eelnevalt loodud funktsooniga makeLearner), teine argument ülesanne (loodud funktsiooniga makeClassifTask, millega eelnevalt defineerisime andmed ja klassi tunnuse).

knnModel <- train(knn, lrTask)

# sisuliselt uusi andmeid meil ei ole, aga on testandmed (objekt test). Saame testandmete peal kontrollida, kui täpselt loodud mudel, mis võtab arvesse 17 lähima erakonna vasak-parempoolsust, erakondi klassifitseerib. (Millised on lähimad, otsustatakse erakondade positsioonide alusel maksu- ja immigratsiooniküsimustes). Klassifitseerimine, milline erakond on vasak- ja milline parempoolne, käib sel juhul funktsiooniga `predict`.

knnProgn <- predict(knnModel, newdata = test)
knnProgn

# uurime mudeli täpsust kokkuvõtlikumalt

table(knnProgn$data$truth, knnProgn$data$response)
descr::crosstab(knnProgn$data$truth, knnProgn$data$response, prop.c = T)

performance(knnProgn, measures = list(mmce, acc))

#### Valideerimine ####

# holdout validation - pmst seda juba tegime, mlr funktsioonid annavad ehk mugavamaid võimalusi (nt klassifitseeriva tunnuse alusel kihistamine)

holdout <- makeResampleDesc(method = "Holdout", split = 2/3, stratify = T)

# defineerime uue ülesande, sest eelnevalt jaotasime ise andmestiku kaheks osaks ja õppeülesandeks valisime ainult ühe osa algsest andmestikust; siin saame ülesandes ära määratleda kogu andmestiku ja selle jaotamine õppe- ja testvalimiks toimub valideerimise käigus

lr_valid <- makeClassifTask(data = df, target = "lrgen")

holdoutvalid <- resample(learner = knn, task = lr_valid, resampling = holdout, measures = list(mmce, acc))

View(holdoutvalid)
holdoutvalid$aggr

calculateConfusionMatrix(holdoutvalid$pred, relative = T)

# k-fold CV

kFold5valid <- resample(learner = knn, task = lr_valid, resampling = cv5, measures = list(mmce, acc))

# eelnevas käsus on argumendi resampling väärtuse cv5 näol tegu sisseehitatud k-fold valideerimise valikuskeemiga, kus andmestik jaotatakse k = 5 osaks, analoogsed on cv2, cv3, cv10. Kui tahame anda k-le muu väärtuse, tuleb eelnevalt defineerida valikuskeem funktsiooniga makeResampleDesc. Valideerimise saab lisaks läbi teha mitu korda, siis on tegu n-ö repeated k-fold CV-ga (iter = folds * reps), vaikeseadena reps = 10.

kFold5_4 <- makeResampleDesc(method = "RepCV", folds = 5, reps = 4, stratify = T)

kFold5_4valid <- resample(learner = knn, task = lr_valid, resampling = kFold5_4, measures = list(mmce, acc))

calculateConfusionMatrix(kFold5_4valid$pred, relative = T)


# LOOCV

LOO <- makeResampleDesc(method = "LOO")

# erakondi ehk andmeridu on suhteliselt vähe, seetõttu ei võta siin LOOCV kuigi kaua aega
LOOvalid <- resample(learner = knn, task = lr_valid, resampling = LOO, measures = list(mmce, acc))

calculateConfusionMatrix(LOOvalid$pred, relative = T)


#### Optimeerimine ####

# Eelnevast mudeli koostamisest saab õppe kontekstis rääkida vaid tinglikult - andsime ise ette k väärtuse ehk mitme lähima naabri liigikuuluvuse alusel indiviid klassifitseeritakse. Püüame nüüd ka mudelit optimeerida ehk leida hüperparameetri väärtus, mis annaks meile mudeli, mis võimaldaks klassikuuluvust prognoosida võimalikult täpselt, kuid samas ei oleks mudel üle sobitatud.

# Defineerime, milliseid k (k nagu kNN, mitte k-fold) väärtusi katsetame
knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 3:200))

# Paneme paika, kuidas just defineeritud k-de hulgast (väärtused 3 kuni 20) erinevaid väärtusi optimeerimisel otsitakse. Antud valik on väga lihtne - proovitakse läbi kõik k väärtused. Kui hüperparameetreid on mitu ja neil kõigil palju erinevaid võimalikke väärtuseid, ei pruugi see mõttekas olla, aga antud juhul on see valik ok.
gridSearch <- makeTuneControlGrid()

# Optimeerime mudelit ehk n-ö tuunime hüperparameetrit - teeme klassifitseerimise läbi, katsetades k (nagu kNN) väärtusi kolmest 267-ni ning valideerime iga k puhul tulemuse k-fold valideerimisega, kus andmestik on jaotatud viieks osaks.

OptimKNN <- tuneParams("classif.knn", task = lr_valid, resampling = cv5, par.set = knnParamSpace, control = gridSearch)

# teeme optimeerimise joonise ka

OptimKNNres <- generateHyperParsEffectData(OptimKNN)

plotHyperParsEffect(OptimKNNres, x = "k", y = "mmce.test.mean", plot.type = "line")

# saame siit teada, et täpseima klassifitseerimistulemuse annab milline k väärtus? 
# Teeme tulemuse põhjal klassifitseerimisprotsessi näitlikult lõpuni läbi ja treenime lõpliku mudeli (kui peame optimaalseks k väärtuseks midagi muud kui see, millel on väikseim mmce väärtus, saab selle järgnevas käsus kirja panna nt k = 45 puhul par.vals = list(k = 45)).

TunedKNN <- setHyperPars(makeLearner("classif.knn"), par.vals = OptimKNN$x)

TunedKnnModel <- train(TunedKNN, lr_valid)
TunedKnnModel

# objektis TunedKnnModel on mudel, mida saaksime edaspidi kasutada uutel andmetel, kus klassikuuluvuse tunnust ei ole, st päris andmetel, kus on erakondade kohta hinnangud maksu- ja immigratsiooniküsimustes ilma teabeta, kas tegu on vasak- või parempoolse erakonnaga. Selleks saab kasutada juba eelnevalt kasutatud funktsiooni predict, kus argumendile newdata omistatakse uus andmestik:

knnProgn <- predict(TunedKnnModel, newdata = ...)